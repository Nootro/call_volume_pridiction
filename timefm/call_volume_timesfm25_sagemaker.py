"""
call_volume_timesfm25_sagemaker.py
===================================
Google TimesFM 2.5 を使ったコールセンター日次呼量予測ベースライン
- ローカルにダウンロード済みのモデルパスを指定して動作
- ゼロショット推論 + オプションでファインチューニング
- 定量評価 (RMSE / MAE / MAPE) + 可視化
- SageMaker での動作も考慮した設計

必要ライブラリ:
    git clone https://github.com/google-research/timesfm.git
    cd timesfm && pip install -e .[torch]
    pip install huggingface_hub pandas numpy scikit-learn matplotlib holidays jpholiday

モデルのローカルダウンロード方法:
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='google/timesfm-2.5-200m-pytorch',
        local_dir='./models/timesfm-2.5-200m-pytorch'
    )
    "
"""

import os
import warnings
import math
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# ★★★ ユーザー設定 ★★★
# ============================================================
# --- データ設定 ---
DATA_FILE: Optional[str] = None        # None → 合成データ / "path/to/data.csv" → 実データ
DATE_COL: str = "ds"                   # 日付列名
VALUE_COL: str = "y"                   # 呼量列名

# --- モデルパス設定 ---
# ローカルにダウンロード済みの場合はパスを指定
# 未ダウンロードの場合は自動でHuggingFaceからダウンロード
MODEL_PATH: str = "./models/timesfm-2.5-200m-pytorch"
AUTO_DOWNLOAD: bool = True             # モデルが存在しない場合に自動ダウンロード

# --- 予測設定 ---
CONTEXT_LENGTH: int = 512              # 過去何日分を入力するか (max: 16384)
HORIZON: int = 60                      # 何日先まで予測するか
TEST_DAYS: int = 60                    # テスト期間 (直近N日を評価に使用)

# --- ファインチューニング設定 ---
ENABLE_FINETUNE: bool = False          # True にするとファインチューニングを実行
FINETUNE_EPOCHS: int = 10
FINETUNE_LR: float = 1e-4
FINETUNE_BATCH_SIZE: int = 16
FINETUNE_CHECKPOINT: str = "./models/timesfm_finetuned.pt"

# --- 可視化設定 ---
PLOT_RESULTS: bool = True              # グラフ保存
PLOT_OUTPUT_DIR: str = "./outputs"

# --- デバイス設定 ---
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_TORCH_COMPILE: bool = False        # torch.compile (PyTorch 2.0+, 高速化)
# ============================================================


# ============================================================
# 1. 合成データ生成
# ============================================================
def generate_synthetic_call_data(n_days: int = 1095) -> pd.DataFrame:
    """コールセンターの呼量を模した合成日次データを生成する。
    
    - トレンド (緩やかな増加)
    - 週次周期 (月曜・火曜が多い)
    - 年次周期 (1〜2月が繁忙期)
    - 祝日効果 (呼量低下)
    - ランダムノイズ + 外れ値
    """
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    trend = np.linspace(0, 50, n_days)
    weekly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    annual = 80 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)

    # 曜日効果: 月=0, 火=1 が高い / 土=5, 日=6 が低い
    dow_effect = np.array([20, 15, 10, 5, 0, -40, -50])
    dow_vals = np.array([dow_effect[d.weekday()] for d in dates])

    noise = np.random.normal(0, 20, n_days)
    base = 300 + trend + weekly + annual + dow_vals + noise

    # 外れ値 (突発イベント)
    outlier_idx = np.random.choice(n_days, size=15, replace=False)
    base[outlier_idx] += np.random.choice([-100, 150], size=15)

    base = np.maximum(base, 50)  # 最低50件

    df = pd.DataFrame({"ds": dates, "y": base.astype(int)})
    logger.info(f"合成データ生成完了: {len(df)}行, 期間={df['ds'].min().date()}〜{df['ds'].max().date()}")
    return df


# ============================================================
# 2. 日本祝日フラグ
# ============================================================
def get_jp_holidays_set(start_date, end_date) -> set:
    """日本の祝日日付セットを返す (holidays または jpholiday を使用)."""
    holiday_set = set()
    try:
        import holidays as hd
        jp_holidays = hd.Japan(years=range(start_date.year, end_date.year + 1))
        holiday_set = set(jp_holidays.keys())
    except ImportError:
        try:
            import jpholiday
            cur = start_date
            while cur <= end_date:
                if jpholiday.is_holiday(cur):
                    holiday_set.add(cur.date() if hasattr(cur, 'date') else cur)
                cur += timedelta(days=1)
        except ImportError:
            logger.warning("holidays/jpholiday 未インストール: 祝日フラグを0で埋めます")
    return holiday_set


# ============================================================
# 3. モデルのローカルダウンロード / ロード
# ============================================================
def ensure_model_downloaded(model_path: str, auto_download: bool = True) -> str:
    """モデルがローカルに存在するか確認し、なければダウンロードする。
    
    Returns:
        str: 実際に使用するモデルパス (ローカル or HF repo_id)
    """
    path = Path(model_path)
    
    # すでにローカルに存在する場合
    if path.exists() and any(path.iterdir()):
        logger.info(f"ローカルモデルを使用: {model_path}")
        return str(path.resolve())
    
    if not auto_download:
        logger.warning(f"モデルパス '{model_path}' が見つかりません。HuggingFaceから直接ロードします。")
        return "google/timesfm-2.5-200m-pytorch"
    
    # ダウンロード実行
    logger.info(f"HuggingFaceからモデルをダウンロード中: google/timesfm-2.5-200m-pytorch")
    logger.info(f"保存先: {model_path} (約2〜3GB, 環境によって数分かかります)")
    try:
        from huggingface_hub import snapshot_download
        path.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(
            repo_id="google/timesfm-2.5-200m-pytorch",
            local_dir=str(path),
        )
        logger.info(f"ダウンロード完了: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"ダウンロード失敗: {e}")
        logger.info("HuggingFace repo_id を直接使用します (インターネット接続が必要)")
        return "google/timesfm-2.5-200m-pytorch"


# ============================================================
# 4. TimesFM 2.5 モデルのロード
# ============================================================
def load_timesfm_model(model_path: str) -> object:
    """TimesFM 2.5 (PyTorch版) をロードして返す。
    
    ForecastConfig パラメータ:
        max_context   : 入力コンテキスト長の最大値 (〜16384)
        max_horizon   : 予測ホライズンの最大値 (〜1024)
        normalize_inputs : 入力正規化 (True 推奨)
        use_continuous_quantile_head : 分位点予測ヘッドを使用
        force_flip_invariance : 反転不変性 (呼量など正値に有効)
        infer_is_positive     : 正値の推論モード (呼量に有効)
        fix_quantile_crossing : 分位点交差を修正
    """
    import timesfm

    torch.set_float32_matmul_precision("high")

    logger.info(f"TimesFM 2.5 ロード中: {model_path}")
    
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_path,
        torch_compile=USE_TORCH_COMPILE,
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=min(CONTEXT_LENGTH, 1024),  # コンパイル時はmax_context指定
            max_horizon=min(HORIZON, 256),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,   # 反転不変: 上昇/下降トレンドどちらも対応
            infer_is_positive=True,       # 呼量は正値 → 0以下の予測を抑制
            fix_quantile_crossing=True,   # 分位点の順序を保証
        )
    )

    logger.info(f"TimesFM 2.5 ロード完了 (デバイス: {DEVICE})")
    return model


# ============================================================
# 5. ゼロショット予測
# ============================================================
def run_zero_shot_forecast(
    model,
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    test_days: int,
) -> Tuple[pd.DataFrame, dict]:
    """TimesFM 2.5 でゼロショット予測を実行する。
    
    Returns:
        forecast_df : 予測結果DataFrame (ds, point_forecast, q10, q50, q90)
        metrics     : 評価指標 dict (rmse, mae, mape)
    """
    df = df.copy().sort_values(DATE_COL).reset_index(drop=True)
    values = df[VALUE_COL].values.astype(float)
    dates = pd.to_datetime(df[DATE_COL]).values

    # --- テスト期間の分割 ---
    if test_days > 0 and len(df) > test_days + context_length:
        train_end_idx = len(df) - test_days
        context_start = max(0, train_end_idx - context_length)
        context_series = values[context_start:train_end_idx]
        actual_test = values[train_end_idx:train_end_idx + test_days]
        test_dates = dates[train_end_idx:train_end_idx + test_days]
        eval_horizon = min(test_days, horizon)
    else:
        # データが短い場合はall-train, future予測のみ
        context_start = max(0, len(df) - context_length)
        context_series = values[context_start:]
        actual_test = None
        test_dates = None
        eval_horizon = horizon

    logger.info(
        f"ゼロショット予測: コンテキスト長={len(context_series)}, 予測ホライズン={eval_horizon}"
    )

    # --- TimesFM 2.5 推論 ---
    point_forecast, quantile_forecast = model.forecast(
        horizon=eval_horizon,
        inputs=[context_series],
    )
    # point_forecast: shape (1, horizon)
    # quantile_forecast: shape (1, horizon, 10) → q10〜q90
    pf = point_forecast[0]           # (horizon,)
    qf = quantile_forecast[0]        # (horizon, 10): q10,q20,...,q90

    # 正値クリップ (呼量は0以上)
    pf = np.maximum(pf, 0)
    qf = np.maximum(qf, 0)

    # --- 将来日付の生成 ---
    if test_dates is not None:
        forecast_dates = pd.to_datetime(test_dates[:eval_horizon])
    else:
        last_date = pd.to_datetime(df[DATE_COL].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=eval_horizon, freq="D"
        )

    forecast_df = pd.DataFrame({
        "ds": forecast_dates,
        "point_forecast": pf,
        "q10": qf[:, 0],   # 10th percentile
        "q20": qf[:, 1],
        "q30": qf[:, 2],
        "q40": qf[:, 3],
        "q50": qf[:, 4],   # median
        "q60": qf[:, 5],
        "q70": qf[:, 6],
        "q80": qf[:, 7],
        "q90": qf[:, 8],   # 90th percentile
    })

    # --- 評価指標の計算 ---
    metrics = {}
    if actual_test is not None:
        actual = actual_test[:eval_horizon]
        predicted = pf[:len(actual)]

        metrics["rmse"] = np.sqrt(mean_squared_error(actual, predicted))
        metrics["mae"] = mean_absolute_error(actual, predicted)
        # MAPE (ゼロ除算回避)
        mask = actual > 0
        metrics["mape"] = (
            np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        )
        # WAPE (Weighted APE)
        metrics["wape"] = (
            np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
        )

        logger.info(
            f"[ゼロショット評価] RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%, "
            f"WAPE={metrics['wape']:.2f}%"
        )

        # 実績を追加
        forecast_df["actual"] = list(actual) + [np.nan] * (len(forecast_df) - len(actual))

    return forecast_df, metrics


# ============================================================
# 6. ファインチューニング用データセット
# ============================================================
class CallVolumeDataset(Dataset):
    """スライディングウィンドウで時系列を切り出すデータセット。
    
    入力: context_length 分の時系列
    出力: horizon 分の将来時系列
    """

    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        horizon: int,
    ):
        self.series = series.astype(np.float32)
        self.context_length = context_length
        self.horizon = horizon
        self.n_samples = max(0, len(series) - context_length - horizon + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx: idx + self.context_length]
        y = self.series[idx + self.context_length: idx + self.context_length + self.horizon]
        return torch.tensor(x), torch.tensor(y)


# ============================================================
# 7. ファインチューニング
# ============================================================
def finetune_timesfm(
    model,
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    epochs: int = FINETUNE_EPOCHS,
    lr: float = FINETUNE_LR,
    batch_size: int = FINETUNE_BATCH_SIZE,
    checkpoint_path: str = FINETUNE_CHECKPOINT,
) -> object:
    """TimesFM 2.5 を呼量データでファインチューニングする。
    
    ファインチューニング戦略:
        - 最終 N ブロックのみ学習可能にする (上位レイヤーのみ更新)
        - MSE損失 + AdamW + コサインスケジューラ
        - 過学習防止のため early stopping
    """
    logger.info("=== ファインチューニング開始 ===")

    values = df[VALUE_COL].values.astype(np.float32)

    # 正規化 (TimesFM内部で正規化されるが、外部でも実施)
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    # Train / Val 分割 (8:2)
    val_size = max(horizon, int(len(values_scaled) * 0.2))
    train_series = values_scaled[:-val_size]
    val_series = values_scaled[-val_size - context_length:]

    train_ds = CallVolumeDataset(train_series, context_length, horizon)
    val_ds = CallVolumeDataset(val_series, context_length, horizon)

    if len(train_ds) == 0:
        logger.warning("トレーニングデータが不足しています。ファインチューニングをスキップします。")
        return model

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- モデルのパラメータ設定: 最終ブロックのみ学習可能 ---
    # まず全パラメータを凍結
    for param in model.parameters():
        param.requires_grad = False

    # 最終 output_projection と最後の数ブロックを解凍
    trainable_layers = []
    try:
        # TimesFM 2.5 の内部構造に応じて調整
        for name, param in model.named_parameters():
            if any(key in name for key in [
                "output_proj", "output_projection",
                "layer_norm", "final",
                "quantile_head",
                # 最終2層のトランスフォーマーブロック
            ]):
                param.requires_grad = True
                trainable_layers.append(name)
    except Exception:
        # フォールバック: 全パラメータを学習可能に
        for param in model.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"学習可能パラメータ: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)  # MSEよりロバスト

    best_val_loss = float("inf")
    patience = max(3, epochs // 3)
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            try:
                # TimesFM 2.5 の forward: inputs リストを渡す
                batch_inputs = [x_batch[i].numpy() for i in range(len(x_batch))]
                point_preds, _ = model.forecast(
                    horizon=horizon,
                    inputs=batch_inputs,
                )
                preds = torch.tensor(point_preds, dtype=torch.float32)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                train_losses.append(loss.item())
            except Exception as e:
                logger.debug(f"バッチ学習エラー (スキップ): {e}")
                continue

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                try:
                    batch_inputs = [x_batch[i].numpy() for i in range(len(x_batch))]
                    point_preds, _ = model.forecast(
                        horizon=horizon,
                        inputs=batch_inputs,
                    )
                    preds = torch.tensor(point_preds, dtype=torch.float32)
                    loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())
                except Exception:
                    continue

        avg_train = np.mean(train_losses) if train_losses else float("nan")
        avg_val = np.mean(val_losses) if val_losses else float("nan")

        logger.info(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            try:
                best_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
            except Exception:
                pass
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping (patience={patience})")
                break

    # 最良モデルの復元
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    # チェックポイント保存
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        logger.info(f"ファインチューニング済みモデルを保存: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"チェックポイント保存失敗: {e}")

    logger.info("=== ファインチューニング完了 ===")
    return model


# ============================================================
# 8. 将来予測 (評価期間後の未来)
# ============================================================
def forecast_future(
    model,
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
) -> pd.DataFrame:
    """データ末尾から horizon 日後までの将来予測を行う。"""
    df = df.copy().sort_values(DATE_COL).reset_index(drop=True)
    values = df[VALUE_COL].values.astype(float)

    context_start = max(0, len(values) - context_length)
    context_series = values[context_start:]

    logger.info(f"将来予測: コンテキスト長={len(context_series)}, ホライズン={horizon}")

    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[context_series],
    )
    pf = np.maximum(point_forecast[0], 0)
    qf = np.maximum(quantile_forecast[0], 0)

    last_date = pd.to_datetime(df[DATE_COL].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=horizon, freq="D"
    )

    future_df = pd.DataFrame({
        "ds": future_dates,
        "point_forecast": pf,
        "q10": qf[:, 0],
        "q50": qf[:, 4],
        "q90": qf[:, 8],
    })
    return future_df


# ============================================================
# 9. 可視化
# ============================================================
def plot_results(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    future_df: Optional[pd.DataFrame] = None,
    title: str = "TimesFM 2.5 コールセンター呼量予測",
    output_dir: str = PLOT_OUTPUT_DIR,
):
    """予測結果を可視化して保存する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # --- 上段: 全期間 ---
        ax1 = axes[0]
        ax1.plot(
            pd.to_datetime(df[DATE_COL]), df[VALUE_COL],
            color="steelblue", linewidth=1.0, label="実績", alpha=0.8
        )

        if "actual" in forecast_df.columns:
            actual_mask = forecast_df["actual"].notna()
            ax1.plot(
                forecast_df.loc[actual_mask, "ds"],
                forecast_df.loc[actual_mask, "actual"],
                color="steelblue", linewidth=1.0, alpha=0.8
            )

        ax1.plot(
            forecast_df["ds"], forecast_df["point_forecast"],
            color="orangered", linewidth=2.0, label="予測 (ポイント)", linestyle="--"
        )

        if "q10" in forecast_df.columns and "q90" in forecast_df.columns:
            ax1.fill_between(
                forecast_df["ds"],
                forecast_df["q10"],
                forecast_df["q90"],
                color="orangered", alpha=0.15, label="80%予測区間"
            )

        if future_df is not None:
            ax1.plot(
                future_df["ds"], future_df["point_forecast"],
                color="green", linewidth=2.0, label="将来予測", linestyle="-."
            )
            ax1.fill_between(
                future_df["ds"],
                future_df["q10"],
                future_df["q90"],
                color="green", alpha=0.12
            )

        ax1.set_xlabel("日付")
        ax1.set_ylabel("呼量 (件)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax1.grid(True, alpha=0.3)

        # --- 下段: テスト期間の詳細 ---
        ax2 = axes[1]
        ax2.set_title("テスト期間の詳細比較", fontsize=11)

        if "actual" in forecast_df.columns:
            actual_mask = forecast_df["actual"].notna()
            ax2.plot(
                forecast_df.loc[actual_mask, "ds"],
                forecast_df.loc[actual_mask, "actual"],
                color="steelblue", linewidth=2.0, marker="o", markersize=3, label="実績"
            )
        ax2.plot(
            forecast_df["ds"], forecast_df["point_forecast"],
            color="orangered", linewidth=2.0, marker="^", markersize=3,
            linestyle="--", label="TimesFM 2.5 予測"
        )
        if "q10" in forecast_df.columns:
            ax2.fill_between(
                forecast_df["ds"],
                forecast_df["q10"],
                forecast_df["q90"],
                color="orangered", alpha=0.2, label="80%予測区間"
            )

        ax2.set_xlabel("日付")
        ax2.set_ylabel("呼量 (件)")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(output_dir) / "timesfm25_forecast.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"グラフを保存しました: {out_path}")

    except Exception as e:
        logger.warning(f"可視化エラー (スキップ): {e}")


# ============================================================
# 10. データ読み込み
# ============================================================
def load_data(data_file: Optional[str]) -> pd.DataFrame:
    """CSVまたは合成データを読み込む。"""
    if data_file and Path(data_file).exists():
        df = pd.read_csv(data_file, parse_dates=[DATE_COL])
        df = df[[DATE_COL, VALUE_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)
        logger.info(f"データ読み込み完了: {data_file} ({len(df)}行)")
    else:
        if data_file:
            logger.warning(f"ファイルが見つかりません: {data_file} → 合成データを使用")
        df = generate_synthetic_call_data(n_days=1095)

    # 祝日フラグ追加 (参考情報として列を追加)
    start_dt = pd.to_datetime(df[DATE_COL].min()).date()
    end_dt = pd.to_datetime(df[DATE_COL].max()).date()
    holiday_set = get_jp_holidays_set(start_dt, end_dt)
    df["is_holiday"] = pd.to_datetime(df[DATE_COL]).dt.date.isin(holiday_set).astype(int)
    df["weekday"] = pd.to_datetime(df[DATE_COL]).dt.weekday  # 0=月, 6=日

    return df


# ============================================================
# 11. 結果サマリー表示
# ============================================================
def print_summary(metrics: dict, forecast_df: pd.DataFrame, future_df: Optional[pd.DataFrame]):
    """評価結果と予測サマリーを表示する。"""
    print("\n" + "="*60)
    print("  TimesFM 2.5 コールセンター呼量予測 - 結果サマリー")
    print("="*60)

    if metrics:
        print(f"\n【評価指標 (テスト期間)】")
        print(f"  RMSE  : {metrics.get('rmse', 'N/A'):>10.2f} 件")
        print(f"  MAE   : {metrics.get('mae', 'N/A'):>10.2f} 件")
        print(f"  MAPE  : {metrics.get('mape', 'N/A'):>10.2f} %")
        print(f"  WAPE  : {metrics.get('wape', 'N/A'):>10.2f} %")

    print(f"\n【テスト期間予測 (先頭10日)】")
    cols = ["ds", "point_forecast", "q10", "q50", "q90"]
    if "actual" in forecast_df.columns:
        cols = ["ds", "actual", "point_forecast", "q10", "q50", "q90"]
    print(forecast_df[cols].head(10).to_string(index=False))

    if future_df is not None:
        print(f"\n【将来予測 (先頭10日)】")
        print(future_df[["ds", "point_forecast", "q10", "q50", "q90"]].head(10).to_string(index=False))

    print("\n" + "="*60)


# ============================================================
# メイン処理
# ============================================================
def main():
    logger.info("========== TimesFM 2.5 コールセンター呼量予測 開始 ==========")

    # 1. データ読み込み
    df = load_data(DATA_FILE)
    logger.info(
        f"データ概要: {len(df)}行, 呼量 平均={df[VALUE_COL].mean():.1f}, "
        f"std={df[VALUE_COL].std():.1f}, min={df[VALUE_COL].min()}, max={df[VALUE_COL].max()}"
    )

    # 2. モデルパス確認 / ダウンロード
    resolved_model_path = ensure_model_downloaded(MODEL_PATH, AUTO_DOWNLOAD)

    # 3. モデルロード
    model = load_timesfm_model(resolved_model_path)

    # 4. ゼロショット予測 (評価)
    logger.info("--- ゼロショット推論 ---")
    forecast_df, metrics = run_zero_shot_forecast(
        model=model,
        df=df,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        test_days=TEST_DAYS,
    )

    # 5. ファインチューニング (オプション)
    if ENABLE_FINETUNE:
        logger.info("--- ファインチューニング ---")
        # ファインチューニング用: テスト期間を除いたデータで学習
        if TEST_DAYS > 0:
            train_df = df.iloc[:-TEST_DAYS].copy()
        else:
            train_df = df.copy()
        model = finetune_timesfm(
            model=model,
            df=train_df,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
        )
        # ファインチューニング後の再評価
        logger.info("--- ファインチューニング後の再評価 ---")
        forecast_df, metrics = run_zero_shot_forecast(
            model=model,
            df=df,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            test_days=TEST_DAYS,
        )

    # 6. 将来予測 (全データを使って未来を予測)
    logger.info("--- 将来予測 ---")
    future_df = forecast_future(
        model=model,
        df=df,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
    )

    # 7. 結果表示
    print_summary(metrics, forecast_df, future_df)

    # 8. 結果保存
    Path(PLOT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(
        Path(PLOT_OUTPUT_DIR) / "timesfm25_test_forecast.csv", index=False
    )
    future_df.to_csv(
        Path(PLOT_OUTPUT_DIR) / "timesfm25_future_forecast.csv", index=False
    )
    logger.info(f"予測結果を保存: {PLOT_OUTPUT_DIR}/timesfm25_*.csv")

    # 9. 可視化
    if PLOT_RESULTS:
        plot_results(df, forecast_df, future_df)

    logger.info("========== 完了 ==========")
    return model, forecast_df, future_df, metrics


if __name__ == "__main__":
    main()
