#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SageMaker 向けに TabM を使用した日次呼量データのマルチステップ予測モデルです。

TabM (Tabular Models) は Yandex が開発した最先端の表形式データ向けモデルで、
パラメータ効率的なアンサンブル学習により高精度な予測を実現します。
このスクリプトは LightGBM 実装の特徴量エンジニアリングを活用しつつ、
TabM の強みである自動アンサンブル学習と効率的な学習を組み合わせます。
"""

import logging
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tabm import TabM
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings
except ImportError as exc:
    raise SystemExit(
        "必要なパッケージが不足しています。以下をインストールしてください:\n"
        "pip install torch tabm rtdl-num-embeddings\n"
    ) from exc

warnings.filterwarnings("ignore")


# -- 祝日生成関数 --
def generate_jp_holidays(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """start から end まで（両端を含む）の日本の祝日の集合を返します。"""
    holidays = set()
    try:
        import holidays as pyholidays  # type: ignore[import]
        jp_holidays = pyholidays.country_holidays(
            "JP", years=list(range(start.year, end.year + 1))
        )
        holidays = {
            pd.Timestamp(day)
            for day in jp_holidays.keys()
            if start <= pd.Timestamp(day) <= end
        }
    except Exception:
        try:
            import jpholiday  # type: ignore[import]
            for date in pd.date_range(start, end, freq="D"):
                if jpholiday.is_holiday(date.to_pydatetime()):
                    holidays.add(date)
        except Exception:
            logging.warning(
                "`holidays` または `jpholiday` のインポートに失敗しました。祝日フラグは 0 となります。"
            )
            holidays = set()
    return holidays


# -- 特徴量エンジニアリング --
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """日付と呼量から各種特徴量を生成します。
    
    TabM は強力な表現学習能力を持つため、豊富な特徴量を提供します。
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    # ラグ特徴量：短期から長期まで
    lag_days = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]
    for lag in lag_days:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    
    # ローリング統計（平均、標準偏差、最小値、最大値）
    windows = [3, 7, 14, 28, 30, 60, 90]
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w, min_periods=1).std()
        df[f"roll_min_{w}"] = df["y"].rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"] = df["y"].rolling(w, min_periods=1).max()
    
    # 指数移動平均（複数スパン）
    for span in [7, 14, 30, 90]:
        df[f"ewm_{span}"] = df["y"].ewm(span=span, adjust=False).mean()
    
    # カレンダー属性
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["ds"].dt.month
    df["day_of_month"] = df["ds"].dt.day
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["ds"].dt.quarter
    df["year"] = df["ds"].dt.year
    
    # 月境界フラグ
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["ds"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["ds"].dt.is_quarter_end.astype(int)
    
    # 週末後の営業日フラグ
    df["is_first_after_weekend"] = (df["ds"].diff().dt.days > 1).astype(int)
    
    # 周期エンコーディング（sin/cos）
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    # 祝日フラグ
    holidays = generate_jp_holidays(df["ds"].min(), df["ds"].max())
    df["is_holiday"] = df["ds"].isin(holidays).astype(int)
    
    # 前祝日・後祝日フラグ
    if holidays:
        pre_holiday_set = {h - pd.Timedelta(days=1) for h in holidays}
        post_holiday_set = {h + pd.Timedelta(days=1) for h in holidays}
        df["is_pre_holiday"] = df["ds"].isin(pre_holiday_set).astype(int)
        df["is_post_holiday"] = df["ds"].isin(post_holiday_set).astype(int)
    else:
        df["is_pre_holiday"] = 0
        df["is_post_holiday"] = 0
    
    # 差分特徴量
    df["diff_1"] = df["y"] - df["lag_1"]
    df["diff_7"] = df["y"] - df["lag_7"]
    df["diff_30"] = df["y"] - df["lag_30"]
    
    # 相対変化率
    for w in [7, 30, 90]:
        roll_mean = df["y"].rolling(w, min_periods=1).mean()
        df[f"rel_change_{w}"] = (df["y"] - roll_mean) / (roll_mean + 1e-8)
    
    # トレンド特徴量（線形回帰の傾き）
    for w in [7, 30]:
        df[f"trend_{w}"] = df["y"].rolling(w, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
    
    # NaN を前方埋めと後方埋めで処理
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    # まだ残っている NaN を 0 で埋める
    df = df.fillna(0)
    
    return df


# -- 時系列交差検証 with TabM --
def time_series_cv_tabm(
    df: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
    use_gpu: bool = False,
) -> Tuple[List[float], TabM, StandardScaler]:
    """時系列データに対して rolling-origin 交差検証を行い、最終モデルを訓練します。

    各フォールドの RMSE スコアと、全データで学習した最終モデルを返します。
    """
    # 特徴量とターゲットの準備
    X = df.drop(columns=["ds", "y"])
    y = df["y"].values
    
    # 標準化（TabM は正規化されたデータで性能が向上）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores: List[float] = []
    
    # デバイス設定
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    
    # TabM のハイパーパラメータ（呼量予測向けにチューニング）
    n_num_features = X.shape[1]
    
    # 各フォールドでモデルを訓練
    fold = 0
    for train_index, valid_index in tscv.split(X_scaled):
        fold += 1
        X_train, X_valid = X_scaled[train_index], X_scaled[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        # TabM モデルを作成（PiecewiseLinearEmbeddings を使用）
        model = TabM.make(
            n_num_features=n_num_features,
            num_embeddings=PiecewiseLinearEmbeddings(
                n_num_features,
                n_bins=64,           # ビン数
                d_embedding=16,      # 埋め込み次元
            ),
            k=32,                    # アンサンブル数
            n_blocks=3,              # ブロック数
            d_block=512,             # ブロックの次元
            dropout=0.1,             # ドロップアウト
            d_out=1,
        ).to(device)
        
        # データローダー
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(device),
            torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(X_valid).to(device),
            torch.FloatTensor(y_valid.reshape(-1, 1)).to(device)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
        
        # オプティマイザーと損失関数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
        criterion = nn.MSELoss()
        
        # 学習ループ
        best_valid_loss = float('inf')
        patience = 15
        patience_counter = 0
        n_epochs = 100
        
        for epoch in range(1, n_epochs + 1):
            # 訓練フェーズ
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                # TabM は (batch_size, k, d_out) の形状で出力
                y_pred = model(X_batch, None)  # cat_cardinalities がないので None
                # k 個の予測を独立に学習（平均損失）
                loss = criterion(y_pred.mean(dim=1), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 検証フェーズ
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    y_pred = model(X_batch, None)
                    # 検証時も k 個の予測を平均
                    loss = criterion(y_pred.mean(dim=1), y_batch)
                    valid_loss += loss.item() * X_batch.size(0)
            
            valid_loss /= len(valid_loader.dataset)
            
            # 早期停止
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # 最終評価
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                y_pred = model(X_batch, None)
                # k 個の予測を平均
                all_preds.append(y_pred.mean(dim=1).cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        y_pred_final = np.concatenate(all_preds, axis=0)
        y_valid_final = np.concatenate(all_targets, axis=0)
        rmse = np.sqrt(mean_squared_error(y_valid_final, y_pred_final))
        rmse_scores.append(rmse)
        print(f"Fold {fold}/{n_splits} RMSE: {rmse:.4f}")
    
    # 全データで最終モデルを訓練
    print("\n全データで最終モデルを訓練中...")
    final_model = TabM.make(
        n_num_features=n_num_features,
        num_embeddings=PiecewiseLinearEmbeddings(
            n_num_features,
            n_bins=64,
            d_embedding=16,
        ),
        k=32,
        n_blocks=3,
        d_block=512,
        dropout=0.1,
        d_out=1,
    ).to(device)
    
    # 検証用に最後の 10% を使用
    split_idx = int(len(X_scaled) * 0.9)
    X_train_final = X_scaled[:split_idx]
    y_train_final = y[:split_idx]
    X_valid_final = X_scaled[split_idx:]
    y_valid_final = y[split_idx:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_final).to(device),
        torch.FloatTensor(y_train_final.reshape(-1, 1)).to(device)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid_final).to(device),
        torch.FloatTensor(y_valid_final.reshape(-1, 1)).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.002, weight_decay=0.0003)
    criterion = nn.MSELoss()
    
    best_valid_loss = float('inf')
    patience = 20
    patience_counter = 0
    n_epochs = 150
    
    for epoch in range(1, n_epochs + 1):
        final_model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = final_model(X_batch, None)
            loss = criterion(y_pred.mean(dim=1), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        final_model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                y_pred = final_model(X_batch, None)
                loss = criterion(y_pred.mean(dim=1), y_batch)
                valid_loss += loss.item() * X_batch.size(0)
        
        valid_loss /= len(valid_loader.dataset)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    return rmse_scores, final_model, scaler


# -- 将来予測 --
def forecast_future(
    df: pd.DataFrame,
    model: TabM,
    scaler: StandardScaler,
    horizon: int,
    device: torch.device,
) -> pd.DataFrame:
    """学習済み TabM モデルを用いて将来の予測を生成します。"""
    future_predictions: List[dict] = []
    current_df = df.copy()
    
    model.eval()
    
    for step in range(horizon):
        # 最終日付から次の日付を計算
        last_date = current_df["ds"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # 特徴量を生成するため、ds と y が NaN の行を追加
        new_row = pd.DataFrame({"ds": [next_date], "y": [np.nan]})
        temp_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # 特徴量を計算し、新しい行を取得
        temp_features = create_features(temp_df)
        last_row = temp_features.iloc[-1:]
        
        X_new = last_row.drop(columns=["ds", "y"]).values
        X_new_scaled = scaler.transform(X_new)
        X_new_tensor = torch.FloatTensor(X_new_scaled).to(device)
        
        # 予測
        with torch.no_grad():
            y_pred = model(X_new_tensor, None)
            # k 個の予測を平均
            y_pred_mean = y_pred.mean(dim=1).cpu().numpy()[0, 0]
        
        future_predictions.append({"ds": next_date, "y_pred": y_pred_mean})
        
        # df を更新（予測値を実際の値として使用）
        current_df = pd.concat(
            [current_df, pd.DataFrame({"ds": [next_date], "y": [y_pred_mean]})],
            ignore_index=True,
        )
        
        if (step + 1) % 10 == 0:
            print(f"予測進捗: {step + 1}/{horizon} 日")
    
    return pd.DataFrame(future_predictions)


# -- 合成データ生成 --
def generate_synthetic_data(n_days: int = 3 * 365) -> pd.DataFrame:
    """複雑なトレンドと季節性を持つ合成データを生成します。"""
    date_range = pd.date_range(
        start=pd.Timestamp.today() - pd.Timedelta(days=n_days), 
        periods=n_days, 
        freq="D"
    )
    
    # 基本トレンド
    trend = np.linspace(100, 200, n_days)
    
    # 週次パターン（平日は高く、週末は低い）
    weekly = 20 * np.sin(2 * np.pi * date_range.dayofweek / 7.0)
    
    # 年次パターン
    annual = 15 * np.sin(2 * np.pi * date_range.dayofyear / 365.0)
    
    # 月次パターン
    monthly = 10 * np.sin(2 * np.pi * date_range.day / 30.0)
    
    # ランダムノイズ
    noise = np.random.normal(scale=8, size=n_days)
    
    # 週末の減少効果
    weekend_effect = -15 * (date_range.dayofweek >= 5).astype(int)
    
    y = trend + weekly + annual + monthly + noise + weekend_effect
    y = np.maximum(y, 0)  # 負の値を避ける
    
    return pd.DataFrame({"ds": date_range, "y": y})


def main() -> None:
    """デフォルト設定でモデルを学習し、将来予測を実行します。"""
    print("=" * 60)
    print("TabM 呼量予測モデル")
    print("=" * 60)
    
    # 設定値（必要に応じて変更してください）
    data_file: Optional[str] = None  # CSV ファイルへのパスを指定
    cv_splits: int = 3                # 交差検証の分割数
    horizon: int = 60                 # 予測したい日数
    random_state: int = 42
    use_gpu: bool = False             # GPU を使用する場合は True

    # データ読み込みまたは生成
    if data_file:
        df_raw = pd.read_csv(data_file)
        if "ds" not in df_raw.columns or "y" not in df_raw.columns:
            raise ValueError("入力 CSV には 'ds' と 'y' の列が必要です。")
    else:
        print("\nデータファイルが指定されていないため、合成データを生成します。")
        df_raw = generate_synthetic_data(n_days=3 * 365)
    
    print(f"データサイズ: {len(df_raw)} 日分")

    # 特徴量エンジニアリング
    print("\n特徴量を生成中...")
    df_features = create_features(df_raw)
    print(f"生成された特徴量数: {len(df_features.columns) - 2} (ds, y を除く)")

    # 時系列交差検証
    print("\n時系列交差検証を実行中...")
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"使用デバイス: {device}")
    
    rmse_scores, final_model, scaler = time_series_cv_tabm(
        df_features, 
        n_splits=cv_splits,
        random_state=random_state,
        use_gpu=use_gpu,
    )
    print(f"\n平均 CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    # 将来予測
    print(f"\n将来 {horizon} 日の予測を生成中...")
    future_df = forecast_future(df_features, final_model, scaler, horizon, device)
    print(f"\nマルチステップ予測 (次の {horizon} 日):")
    print(future_df.head(20))
    
    # 統計サマリー
    print(f"\n予測統計:")
    print(f"  平均: {future_df['y_pred'].mean():.2f}")
    print(f"  標準偏差: {future_df['y_pred'].std():.2f}")
    print(f"  最小値: {future_df['y_pred'].min():.2f}")
    print(f"  最大値: {future_df['y_pred'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("TabM の特徴:")
    print("  ✓ パラメータ効率的なアンサンブル学習")
    print("  ✓ 自動で k=32 個のモデルを並列学習")
    print("  ✓ 重み共有による高速化と正則化")
    print("  ✓ PiecewiseLinearEmbeddings で高精度")
    print("=" * 60)


if __name__ == "__main__":
    main()
