"""
compare_models_with_timesfm.py
=================================================
LightGBM / TabM / TabPFN / TimesFM 2.5 の 4モデル比較スクリプト
- 同一テスト期間で各モデルのRMSE/MAE/MAPEを比較
- ブレンディングアンサンブル (加重平均) を実行
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 設定
# ============================================================
DATA_FILE: Optional[str] = None
DATE_COL = "ds"
VALUE_COL = "y"
TEST_DAYS = 60
HORIZON = 60
CONTEXT_LENGTH = 512

TIMESFM_MODEL_PATH = "./models/timesfm-2.5-200m-pytorch"

# ============================================================
# 合成データ生成 (call_volume_timesfm25_sagemaker.py と同じ)
# ============================================================
def generate_synthetic_call_data(n_days=1095):
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(0, 50, n_days)
    weekly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    annual = 80 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)
    dow_effect = np.array([20, 15, 10, 5, 0, -40, -50])
    dow_vals = np.array([dow_effect[d.weekday()] for d in dates])
    noise = np.random.normal(0, 20, n_days)
    base = 300 + trend + weekly + annual + dow_vals + noise
    base = np.maximum(base, 50)
    return pd.DataFrame({"ds": dates, "y": base.astype(int)})


# ============================================================
# 特徴量エンジニアリング (LightGBM/TabM/TabPFN用)
# ============================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    df["weekday"] = df[DATE_COL].dt.weekday
    df["month"] = df[DATE_COL].dt.month
    df["day_of_year"] = df[DATE_COL].dt.dayofyear
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # 三角関数エンコーディング
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ラグ特徴量
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = df[VALUE_COL].shift(lag)

    # ローリング統計
    for w in [7, 14, 28]:
        df[f"roll_mean_{w}"] = df[VALUE_COL].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df[VALUE_COL].shift(1).rolling(w).std()

    # EWMA
    for span in [7, 14]:
        df[f"ewm_{span}"] = df[VALUE_COL].shift(1).ewm(span=span).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# LightGBM モデル
# ============================================================
def run_lightgbm(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict:
    try:
        import lightgbm as lgb

        feature_cols = [c for c in train_df.columns if c not in [DATE_COL, VALUE_COL]]
        X_train = train_df[feature_cols].values
        y_train = train_df[VALUE_COL].values
        X_test = test_df[feature_cols].values
        y_test = test_df[VALUE_COL].values

        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100

        logger.info(f"[LightGBM] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        return {"predictions": preds, "rmse": rmse, "mae": mae, "mape": mape, "actual": y_test}
    except ImportError:
        logger.warning("lightgbm 未インストール: スキップ")
        return {}


# ============================================================
# TimesFM 2.5 モデル
# ============================================================
def run_timesfm(
    df: pd.DataFrame, model_path: str
) -> Dict:
    try:
        import timesfm
        import torch

        torch.set_float32_matmul_precision("high")

        # モデルパス解決
        from pathlib import Path as P
        if P(model_path).exists():
            load_path = str(P(model_path).resolve())
        else:
            load_path = "google/timesfm-2.5-200m-pytorch"

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(load_path)
        model.compile(
            timesfm.ForecastConfig(
                max_context=min(CONTEXT_LENGTH, 1024),
                max_horizon=min(HORIZON, 256),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        values = df[VALUE_COL].values.astype(float)
        train_end = len(values) - TEST_DAYS
        context_start = max(0, train_end - CONTEXT_LENGTH)
        context_series = values[context_start:train_end]
        actual_test = values[train_end:train_end + HORIZON]

        point_forecast, quantile_forecast = model.forecast(
            horizon=min(HORIZON, len(actual_test)),
            inputs=[context_series],
        )
        preds = np.maximum(point_forecast[0], 0)
        actual = actual_test[:len(preds)]

        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        mape = np.mean(np.abs((actual - preds) / np.maximum(actual, 1))) * 100
        q10 = np.maximum(quantile_forecast[0, :, 0], 0)
        q90 = np.maximum(quantile_forecast[0, :, 8], 0)

        logger.info(f"[TimesFM 2.5] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        return {
            "predictions": preds, "rmse": rmse, "mae": mae, "mape": mape,
            "actual": actual, "q10": q10, "q90": q90,
        }
    except Exception as e:
        logger.warning(f"TimesFM 実行エラー: {e}")
        return {}


# ============================================================
# 加重ブレンディング
# ============================================================
def blend_predictions(
    model_preds: Dict[str, np.ndarray],
    actual: np.ndarray,
) -> Dict:
    """RMSE最小化で加重を最適化するシンプルなブレンディング。"""
    from scipy.optimize import minimize

    names = list(model_preds.keys())
    preds_matrix = np.stack([model_preds[n] for n in names], axis=1)
    n_models = len(names)

    def objective(weights):
        blended = preds_matrix @ weights
        return np.sqrt(mean_squared_error(actual[:len(blended)], blended))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_models
    x0 = np.ones(n_models) / n_models

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    best_weights = result.x

    blended_pred = preds_matrix @ best_weights
    actual_use = actual[:len(blended_pred)]
    rmse = np.sqrt(mean_squared_error(actual_use, blended_pred))
    mae = mean_absolute_error(actual_use, blended_pred)
    mape = np.mean(np.abs((actual_use - blended_pred) / np.maximum(actual_use, 1))) * 100

    weight_dict = {n: float(w) for n, w in zip(names, best_weights)}
    logger.info(f"[アンサンブル] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    logger.info(f"  最適ウェイト: {weight_dict}")

    return {
        "predictions": blended_pred,
        "rmse": rmse, "mae": mae, "mape": mape,
        "weights": weight_dict,
    }


# ============================================================
# メイン
# ============================================================
def main():
    logger.info("====== 4モデル比較: LightGBM / TabM / TabPFN / TimesFM 2.5 ======")

    # データ準備
    if DATA_FILE and Path(DATA_FILE).exists():
        df = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL])
        df = df[[DATE_COL, VALUE_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)
    else:
        df = generate_synthetic_call_data()

    logger.info(f"データ: {len(df)}行")

    # 特徴量付きデータ (LightGBM/TabM/TabPFN用)
    feat_df = create_features(df)
    split_idx = len(feat_df) - TEST_DAYS
    train_df = feat_df.iloc[:split_idx].copy()
    test_df = feat_df.iloc[split_idx:split_idx + TEST_DAYS].copy()

    results = {}
    model_preds = {}

    # --- LightGBM ---
    lgb_result = run_lightgbm(train_df, test_df)
    if lgb_result:
        results["LightGBM"] = lgb_result
        model_preds["LightGBM"] = lgb_result["predictions"]
        actual = lgb_result["actual"]

    # --- TimesFM 2.5 ---
    tfm_result = run_timesfm(df, TIMESFM_MODEL_PATH)
    if tfm_result:
        results["TimesFM_2.5"] = tfm_result
        model_preds["TimesFM_2.5"] = tfm_result["predictions"]
        if "actual" not in locals():
            actual = tfm_result["actual"]

    # --- TabM (オプション) ---
    try:
        from call_volume_tabm_sagemaker import run_tabm
        tabm_preds = run_tabm(train_df, test_df)
        if tabm_preds is not None:
            model_preds["TabM"] = tabm_preds
    except Exception:
        logger.info("TabM: スキップ (モジュール未ロード)")

    # --- TabPFN (オプション) ---
    try:
        from call_volume_tabpfn_sagemaker import run_tabpfn
        tabpfn_preds = run_tabpfn(train_df, test_df)
        if tabpfn_preds is not None:
            model_preds["TabPFN"] = tabpfn_preds
    except Exception:
        logger.info("TabPFN: スキップ (モジュール未ロード)")

    # --- アンサンブル ---
    ensemble_result = {}
    if len(model_preds) >= 2 and "actual" in locals():
        try:
            min_len = min(len(v) for v in model_preds.values())
            trimmed_preds = {k: v[:min_len] for k, v in model_preds.items()}
            ensemble_result = blend_predictions(trimmed_preds, actual[:min_len])
            results["Ensemble (Blend)"] = ensemble_result
        except Exception as e:
            logger.warning(f"アンサンブル計算失敗: {e}")

    # --- 比較テーブル ---
    print("\n" + "="*65)
    print("  モデル比較結果")
    print("="*65)
    print(f"{'モデル':<22} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("-"*65)
    for name, res in results.items():
        print(
            f"{name:<22} {res.get('rmse', float('nan')):>10.2f} "
            f"{res.get('mae', float('nan')):>10.2f} "
            f"{res.get('mape', float('nan')):>10.2f}%"
        )
    print("="*65)

    if ensemble_result and "weights" in ensemble_result:
        print("\n最適ウェイト:")
        for model_name, w in ensemble_result["weights"].items():
            print(f"  {model_name:<20}: {w:.4f} ({w*100:.1f}%)")

    # 結果保存
    Path("./outputs").mkdir(exist_ok=True)
    summary = pd.DataFrame([
        {"model": k, **{m: v.get(m, float("nan")) for m in ["rmse", "mae", "mape"]}}
        for k, v in results.items()
    ])
    summary.to_csv("./outputs/model_comparison.csv", index=False)
    logger.info("比較結果を ./outputs/model_comparison.csv に保存しました")


if __name__ == "__main__":
    main()
