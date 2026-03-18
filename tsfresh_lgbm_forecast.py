#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
tsfresh + LightGBM 時系列予測スクリプト
=============================================================================
概要:
  - ds, y 形式（Prophet形式）のCSVを読み込み
  - tsfresh でローリング窓特徴量を自動生成
  - Optuna でLightGBMのハイパーパラメータを最適化
  - テスト期間 = データ末尾2ヶ月（カレンダー月で厳密に計算）
  - 評価指標: MAPE, RMSE, MAE（全体・1ヶ月目・2ヶ月目）
  - Feature Importance (Gain / Split) + SHAP 分析
  - 【追加】全データを使ったカレンダー月2ヶ月先の未来予測
      再帰予測（ラグ特徴量を予測値で逐次更新）
      tsfresh特徴量は実データ窓を一括抽出してから補完

使用方法:
  python tsfresh_lgbm_forecast.py --csv data.csv
  python tsfresh_lgbm_forecast.py --csv data.csv --window 60 --trials 30 --fast

依存ライブラリ:
  pip install tsfresh lightgbm optuna shap matplotlib seaborn pandas numpy scikit-learn
=============================================================================
"""

import argparse
import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import timedelta
import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

from tsfresh import extract_features
from tsfresh.feature_extraction import (
    EfficientFCParameters,
    MinimalFCParameters,
    ComprehensiveFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_WINDOW  = 90       # tsfresh ローリング窓サイズ（日数）
DEFAULT_TRIALS  = 50       # Optuna 試行回数
DEFAULT_CV_FOLDS = 3       # TimeSeriesSplit のfold数
OUTPUT_DIR      = "tsfresh_lgbm_output"  # 出力フォルダ
RANDOM_SEED     = 42

# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred, eps=1e-8):
    """Symmetric MAPE"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_metrics(y_true, y_pred, label=""):
    """評価指標を計算してdictで返す"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    metrics = {
        "MAPE (%)":  round(mape(y_true, y_pred), 4),
        "SMAPE (%)": round(smape(y_true, y_pred), 4),
        "RMSE":      round(rmse(y_true, y_pred), 4),
        "MAE":       round(mae(y_true, y_pred), 4),
        "MBE":       round(float(np.mean(y_pred - y_true)), 4),   # Mean Bias Error
        "R2":        round(float(1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)), 4),
    }
    if label:
        logger.info(f"\n{'='*55}")
        logger.info(f"  評価指標: {label}")
        logger.info(f"{'='*55}")
        for k, v in metrics.items():
            logger.info(f"  {k:<15}: {v}")
    return metrics

# ──────────────────────────────────────────────────────────────────────────────
# 1. データ読み込み・前処理
# ──────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"データ読み込み: {csv_path}")
    df = pd.read_csv(csv_path)

    # ds列を日付に変換
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("CSVに 'ds' と 'y' 列が必要です")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # 欠損値補完（線形補間）
    n_nan = df["y"].isna().sum()
    if n_nan > 0:
        logger.warning(f"y列に欠損値 {n_nan} 件 → 線形補間")
        df["y"] = df["y"].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    logger.info(f"データ期間: {df['ds'].min().date()} ～ {df['ds'].max().date()}")
    logger.info(f"行数: {len(df)}")
    logger.info(f"y の統計: mean={df['y'].mean():.2f}, std={df['y'].std():.2f}, "
                f"min={df['y'].min():.2f}, max={df['y'].max():.2f}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. テスト期間をカレンダー月で厳密に計算
# ──────────────────────────────────────────────────────────────────────────────
def get_test_split_dates(df: pd.DataFrame):
    """
    末尾2ヶ月をカレンダー月で厳密に計算

    例: データ最終日 2025/02/28
        month2: 2025/02/01 ~ 2025/02/28  （第2テスト月 = 最終月）
        month1: 2025/01/01 ~ 2025/01/31  （第1テスト月 = その前月）
        test  : 2025/01/01 ~ 2025/02/28
        train : ~ 2024/12/31
    """
    last_date = df["ds"].max()

    # ── 第2テスト月（最終月）──────────────────────────────
    m2_year  = last_date.year
    m2_month = last_date.month
    m2_start = pd.Timestamp(m2_year, m2_month, 1)
    m2_end   = last_date  # データの最終日（月末とは限らない）

    # ── 第1テスト月（その前月）───────────────────────────
    if m2_month == 1:
        m1_year  = m2_year - 1
        m1_month = 12
    else:
        m1_year  = m2_year
        m1_month = m2_month - 1
    m1_start = pd.Timestamp(m1_year, m1_month, 1)
    last_day_of_m1 = calendar.monthrange(m1_year, m1_month)[1]
    m1_end   = pd.Timestamp(m1_year, m1_month, last_day_of_m1)

    # ── テスト全体 ────────────────────────────────────────
    test_start = m1_start
    test_end   = m2_end
    train_end  = m1_start - pd.Timedelta(days=1)

    logger.info(f"\n{'='*55}")
    logger.info(f"  テスト期間（カレンダー月）")
    logger.info(f"  Month1 (第1テスト月): {m1_start.date()} ～ {m1_end.date()}")
    logger.info(f"  Month2 (第2テスト月): {m2_start.date()} ～ {m2_end.date()}")
    logger.info(f"  テスト全体          : {test_start.date()} ～ {test_end.date()}")
    logger.info(f"  訓練期間            : ～ {train_end.date()}")
    logger.info(f"{'='*55}")

    return {
        "train_end":  train_end,
        "test_start": test_start,
        "test_end":   test_end,
        "m1_start":   m1_start,
        "m1_end":     m1_end,
        "m2_start":   m2_start,
        "m2_end":     m2_end,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. tsfresh ローリング特徴量生成
# ──────────────────────────────────────────────────────────────────────────────
def build_tsfresh_features(df: pd.DataFrame, window_size: int, fast: bool = False) -> pd.DataFrame:
    """
    各日付 t に対して、過去 window_size 日分の y 値から
    tsfresh特徴量を抽出し、横断面特徴量行列を返す。

    Parameters
    ----------
    df          : ds, y のDataFrame（ソート済み）
    window_size : ローリング窓サイズ（日数）
    fast        : True → MinimalFCParameters（高速・少特徴量）
                  False → EfficientFCParameters（中程度）

    Returns
    -------
    features_df : index = ds, columns = tsfresh特徴量
    """
    logger.info(f"tsfresh特徴量を生成中 (window={window_size}, fast={fast}) ...")
    logger.info(f"  FeatureSettings: {'Minimal' if fast else 'Efficient'}")

    fc_params = MinimalFCParameters() if fast else EfficientFCParameters()

    # ── ローリング窓でサンプル生成 ─────────────────────────────────────────
    records = []
    dates   = df["ds"].values
    values  = df["y"].values
    n       = len(df)

    for i in range(window_size, n):
        # 過去 window_size 日分の y（iの日付を予測対象としてi-window_size~i-1を使用）
        window_vals = values[i - window_size : i]
        window_times = np.arange(window_size)  # 0,1,...,W-1 の相対時刻
        pred_date = dates[i]

        for t, v in zip(window_times, window_vals):
            records.append({
                "id":    pred_date,   # 予測対象日がID
                "time":  t,
                "value": v,
            })

    rolled_df = pd.DataFrame(records)
    total_ids = rolled_df["id"].nunique()
    logger.info(f"  ローリング窓サンプル数: {total_ids} （窓サイズ {window_size}日）")

    # ── tsfresh 特徴量抽出 ─────────────────────────────────────────────────
    logger.info("  tsfresh extract_features を実行中（しばらくお待ちください）...")
    features = extract_features(
        rolled_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_params,
        n_jobs=1,          # 並列数（環境に応じて増やす）
        show_warnings=False,
        disable_progressbar=False,
    )

    # ── NaN処理（tsfreshのimpute） ──────────────────────────────────────────
    logger.info("  NaN補完（tsfresh impute）中...")
    impute(features)

    # ── 定数列除去 ──────────────────────────────────────────────────────────
    const_cols = [c for c in features.columns if features[c].nunique() <= 1]
    if const_cols:
        logger.info(f"  定数列 {len(const_cols)} 件を除去")
        features = features.drop(columns=const_cols)

    features.index = pd.to_datetime(features.index)
    features.index.name = "ds"

    logger.info(f"  ✅ tsfresh特徴量: {features.shape[1]} 個 × {len(features)} サンプル")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# 4. 補助特徴量（カレンダー・ラグ）
# ──────────────────────────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """カレンダー特徴量を追加"""
    d = df["ds"]
    df["dow"]          = d.dt.dayofweek
    df["month"]        = d.dt.month
    df["day"]          = d.dt.day
    df["dayofyear"]    = d.dt.dayofyear
    df["week"]         = d.dt.isocalendar().week.astype(int)
    df["quarter"]      = d.dt.quarter
    df["is_weekend"]   = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = (d.dt.day <= 3).astype(int)
    df["is_month_end"]   = (d.dt.day >= 28).astype(int)
    # 循環エンコーディング
    df["sin_dow"]      = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]      = np.cos(2 * np.pi * df["dow"] / 7)
    df["sin_month"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"]      = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"]      = np.cos(2 * np.pi * df["dayofyear"] / 365)
    # jpholiday（インストール済みの場合）
    try:
        import jpholiday
        df["is_holiday"]     = d.apply(lambda x: int(jpholiday.is_holiday(x)))
        df["is_golden_week"] = d.apply(
            lambda x: int((x.month == 4 and x.day >= 29) or (x.month == 5 and x.day <= 5))
        )
        df["is_obon"]        = d.apply(lambda x: int(x.month == 8 and 13 <= x.day <= 15))
        df["is_year_end_ny"] = d.apply(
            lambda x: int((x.month == 12 and x.day >= 28) or (x.month == 1 and x.day <= 4))
        )
        logger.info("  ✅ jpholiday 祝日特徴量を追加")
    except ImportError:
        logger.warning("  ⚠️ jpholiday が見つかりません。祝日特徴量をスキップ")
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3, 7, 14, 21, 28)) -> pd.DataFrame:
    """ラグ特徴量を追加"""
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Optuna によるハイパーパラメータ最適化
# ──────────────────────────────────────────────────────────────────────────────
def optuna_optimize(X_train: pd.DataFrame, y_train: pd.Series,
                    n_trials: int, cv_folds: int) -> dict:
    """TimeSeriesSplit + Optuna で LightGBM パラメータを最適化"""
    logger.info(f"\n{'='*55}")
    logger.info(f"  Optuna ハイパーパラメータ最適化")
    logger.info(f"  試行回数: {n_trials}, CV分割: {cv_folds}")
    logger.info(f"{'='*55}")

    tscv = TimeSeriesSplit(n_splits=cv_folds, gap=0)

    def objective(trial):
        params = {
            "objective":       "regression",
            "metric":          "rmse",
            "verbosity":       -1,
            "boosting_type":   trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves":      trial.suggest_int("num_leaves", 16, 256),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":    trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":  trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":  trial.suggest_float("min_split_gain", 0.0, 1.0),
            "max_bin":         trial.suggest_int("max_bin", 63, 511),
            "random_state":    RANDOM_SEED,
        }

        fold_scores = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            X_tr  = X_train.iloc[fold_train_idx]
            y_tr  = y_train.iloc[fold_train_idx]
            X_val = X_train.iloc[fold_val_idx]
            y_val = y_train.iloc[fold_val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            pred = model.predict(X_val)
            fold_scores.append(rmse(y_val, pred))

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params.update({
        "objective":    "regression",
        "metric":       "rmse",
        "verbosity":    -1,
        "random_state": RANDOM_SEED,
    })

    logger.info(f"  ✅ 最良RMSE（CV平均）: {study.best_value:.4f}")
    logger.info(f"  ✅ 最良パラメータ: {study.best_params}")

    return best_params, study


# ──────────────────────────────────────────────────────────────────────────────
# 6. 最終モデル訓練・予測
# ──────────────────────────────────────────────────────────────────────────────
def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """最終モデルを訓練してLGBM modelを返す"""
    logger.info("\n最終モデルを訓練中...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(100)],
    )
    logger.info(f"  ✅ 最終モデル訓練完了 (best_iteration={model.best_iteration_})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 7. 可視化
# ──────────────────────────────────────────────────────────────────────────────
def plot_forecast(df_test: pd.DataFrame, df_train: pd.DataFrame,
                  split_dates: dict, output_dir: str):
    """予測vs実績の時系列グラフ"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── (上) 全期間 ────────────────────────────────────────────────
    ax = axes[0]
    # 訓練データの最後90日
    show_train = df_train.tail(90)
    ax.plot(show_train["ds"], show_train["y"],
            color="steelblue", label="訓練データ（直近90日）", linewidth=1.5, alpha=0.7)
    ax.plot(df_test["ds"], df_test["y"],
            color="black", label="実績（テスト）", linewidth=2)
    ax.plot(df_test["ds"], df_test["y_pred"],
            color="crimson", label="予測", linewidth=2, linestyle="--")

    # 月境界線
    ax.axvline(split_dates["m1_start"], color="orange", linestyle=":", linewidth=1.5,
               label=f"Month1開始 ({split_dates['m1_start'].date()})")
    ax.axvline(split_dates["m2_start"], color="purple", linestyle=":", linewidth=1.5,
               label=f"Month2開始 ({split_dates['m2_start'].date()})")

    ax.set_title("予測 vs 実績（全体）", fontsize=14, fontweight="bold")
    ax.set_xlabel("日付")
    ax.set_ylabel("呼量 (y)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)

    # ── (下) テスト期間ズーム ────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(df_test["ds"],
                     df_test["y"] - df_test["residual"].abs(),
                     df_test["y"] + df_test["residual"].abs(),
                     alpha=0.15, color="crimson", label="残差帯")
    ax2.plot(df_test["ds"], df_test["y"],
             color="black", label="実績", linewidth=2.5, marker="o", markersize=4)
    ax2.plot(df_test["ds"], df_test["y_pred"],
             color="crimson", label="予測", linewidth=2.5, linestyle="--", marker="^", markersize=4)

    m1_data = df_test[df_test["period"] == "Month1"]
    m2_data = df_test[df_test["period"] == "Month2"]
    ax2.axvspan(split_dates["m1_start"], split_dates["m1_end"],
                alpha=0.08, color="orange", label="Month1")
    ax2.axvspan(split_dates["m2_start"], split_dates["m2_end"],
                alpha=0.08, color="purple", label="Month2")

    ax2.set_title("予測 vs 実績（テスト期間ズーム）", fontsize=14, fontweight="bold")
    ax2.set_xlabel("日付")
    ax2.set_ylabel("呼量 (y)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_forecast.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 予測グラフ保存: {save_path}")


def plot_metrics_table(metrics_dict: dict, output_dir: str):
    """評価指標テーブルをグラフとして保存"""
    periods = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())

    data = np.array([[metrics_dict[p][m] for m in metric_names] for p in periods])
    df_table = pd.DataFrame(data, index=periods, columns=metric_names)

    fig, ax = plt.subplots(figsize=(12, max(3, len(periods) + 1)))
    ax.axis("off")

    colors_header = [["#1A2E4A"] * len(metric_names)]
    cell_colors = []
    for i, period in enumerate(periods):
        row_colors = []
        for j, m in enumerate(metric_names):
            if "MAPE" in m or "SMAPE" in m:
                val = data[i, j]
                # MAPE: 5%未満=緑、10%未満=黄、それ以上=赤
                if val < 5:
                    c = "#D1FAE5"
                elif val < 10:
                    c = "#FEF3C7"
                else:
                    c = "#FEE2E2"
            elif m == "R2":
                val = data[i, j]
                if val > 0.9:
                    c = "#D1FAE5"
                elif val > 0.7:
                    c = "#FEF3C7"
                else:
                    c = "#FEE2E2"
            else:
                c = "#F9FAFB" if i % 2 == 0 else "#FFFFFF"
            row_colors.append(c)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=data,
        rowLabels=periods,
        colLabels=metric_names,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.2)

    # ヘッダー色設定
    for j in range(len(metric_names)):
        table[(0, j)].set_facecolor("#1A2E4A")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(len(periods)):
        table[(i + 1, -1)].set_facecolor("#2563EB")
        table[(i + 1, -1)].set_text_props(color="white", fontweight="bold")

    ax.set_title("評価指標サマリー", fontsize=16, fontweight="bold", pad=30,
                 color="#1A2E4A")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_metrics_table.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 評価指標テーブル保存: {save_path}")


def plot_monthly_metrics(metrics_dict: dict, output_dir: str):
    """月別評価指標の棒グラフ"""
    periods = [k for k in metrics_dict.keys() if k != "Test_All"]
    if len(periods) < 2:
        return

    metrics_to_plot = ["MAPE (%)", "SMAPE (%)", "RMSE", "MAE"]
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#3B82F6", "#8B5CF6", "#10B981"]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [metrics_dict[p].get(metric, np.nan) for p in periods]
        bars = ax.bar(periods, vals, color=colors[:len(periods)], edgecolor="white",
                      linewidth=1.5, width=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_facecolor("#F9FAFB")

    fig.suptitle("月別評価指標比較", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_monthly_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 月別指標グラフ保存: {save_path}")


def plot_feature_importance(model, feature_names: list, output_dir: str, top_n: int = 30):
    """Feature Importance (Gain + Split) を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n // 2)))

    for ax, imp_type in zip(axes, ["gain", "split"]):
        imp = model.booster_.feature_importance(importance_type=imp_type)
        imp_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)

        bars = ax.barh(
            imp_df["feature"][::-1],
            imp_df["importance"][::-1],
            color=plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_df)))[::-1],
            edgecolor="white",
        )
        ax.set_title(f"Feature Importance ({imp_type.upper()})\nTop {top_n}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_facecolor("#F9FAFB")

    plt.suptitle("LightGBM Feature Importance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 Feature Importance保存: {save_path}")

    # CSV保存
    gain_df = pd.DataFrame({
        "feature":    feature_names,
        "gain":       model.booster_.feature_importance(importance_type="gain"),
        "split":      model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    csv_path = os.path.join(output_dir, "feature_importance.csv")
    gain_df.to_csv(csv_path, index=False)
    logger.info(f"  📊 Feature Importance CSV保存: {csv_path}")


def plot_shap(model, X_test: pd.DataFrame, output_dir: str, top_n: int = 30):
    """SHAP分析の可視化"""
    logger.info("  SHAP値を計算中...")
    explainer = shap.TreeExplainer(model)

    # テストデータが大きい場合はサンプリング
    X_shap = X_test.iloc[:min(200, len(X_test))]
    shap_values = explainer.shap_values(X_shap)

    # ── (1) Summary Plot（Bee Swarm） ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, top_n // 2)))
    shap.summary_plot(
        shap_values, X_shap,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary Plot（Top {top_n}）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 SHAP Summary保存: {save_path}")

    # ── (2) Bar Plot（平均絶対SHAP値） ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, top_n // 2)))
    shap.summary_plot(
        shap_values, X_shap,
        max_display=top_n,
        show=False,
        plot_type="bar",
    )
    plt.title(f"SHAP Feature Importance（平均|SHAP|, Top {top_n}）",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "06_shap_bar.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 SHAP Barplot保存: {save_path2}")

    # ── SHAP値CSV保存 ───────────────────────────────────────────────────────
    shap_mean = pd.DataFrame({
        "feature":    X_shap.columns.tolist(),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_csv = os.path.join(output_dir, "shap_importance.csv")
    shap_mean.to_csv(shap_csv, index=False)
    logger.info(f"  📊 SHAP CSV保存: {shap_csv}")


def plot_residuals(df_test: pd.DataFrame, output_dir: str):
    """残差分析"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 残差時系列
    ax = axes[0]
    ax.plot(df_test["ds"], df_test["residual"], color="steelblue",
            linewidth=1.5, marker="o", markersize=3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.fill_between(df_test["ds"], df_test["residual"], 0,
                    where=(df_test["residual"] > 0), alpha=0.3, color="red", label="過大予測")
    ax.fill_between(df_test["ds"], df_test["residual"], 0,
                    where=(df_test["residual"] < 0), alpha=0.3, color="blue", label="過小予測")
    ax.set_title("残差（実績 - 予測）", fontsize=12, fontweight="bold")
    ax.set_xlabel("日付")
    ax.set_ylabel("残差")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 残差ヒストグラム
    ax2 = axes[1]
    ax2.hist(df_test["residual"], bins=20, color="steelblue", edgecolor="white",
             alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.axvline(df_test["residual"].mean(), color="orange", linestyle="--",
                linewidth=1.5, label=f"平均={df_test['residual'].mean():.2f}")
    ax2.set_title("残差分布", fontsize=12, fontweight="bold")
    ax2.set_xlabel("残差")
    ax2.set_ylabel("頻度")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 予測 vs 実績の散布図
    ax3 = axes[2]
    ax3.scatter(df_test["y"], df_test["y_pred"],
                alpha=0.6, color="steelblue", edgecolor="white", s=50)
    lim_min = min(df_test["y"].min(), df_test["y_pred"].min()) * 0.95
    lim_max = max(df_test["y"].max(), df_test["y_pred"].max()) * 1.05
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="完全一致線")
    ax3.set_title("実績 vs 予測（散布図）", fontsize=12, fontweight="bold")
    ax3.set_xlabel("実績")
    ax3.set_ylabel("予測")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle("残差分析", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "07_residuals.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 残差分析保存: {save_path}")


def plot_optuna_results(study, output_dir: str):
    """Optuna最適化の可視化"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 最適化履歴
        ax = axes[0]
        trials_df = study.trials_dataframe()
        ax.plot(trials_df.index, trials_df["value"],
                "o-", color="steelblue", markersize=4, linewidth=1.5, alpha=0.7)
        ax.axhline(study.best_value, color="red", linestyle="--",
                   linewidth=1.5, label=f"Best: {study.best_value:.4f}")
        ax.set_title("Optuna 最適化履歴（RMSE）", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial番号")
        ax.set_ylabel("RMSE（CV平均）")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # パラメータ重要度
        ax2 = axes[1]
        try:
            param_importances = optuna.importance.get_param_importances(study)
            params = list(param_importances.keys())[:10]
            importances = [param_importances[p] for p in params]
            ax2.barh(params[::-1], importances[::-1],
                     color=plt.cm.Blues(np.linspace(0.4, 0.9, len(params)))[::-1])
            ax2.set_title("Optunaパラメータ重要度", fontsize=12, fontweight="bold")
            ax2.set_xlabel("重要度")
            ax2.grid(True, axis="x", alpha=0.3)
        except Exception:
            ax2.text(0.5, 0.5, "パラメータ重要度の計算に\n失敗しました",
                     ha="center", va="center", transform=ax2.transAxes)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "08_optuna_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  📊 Optuna結果保存: {save_path}")
    except Exception as e:
        logger.warning(f"  Optunaグラフ生成エラー: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 8-EX. 未来予測（全データ使用・カレンダー月2ヶ月先）
# ──────────────────────────────────────────────────────────────────────────────
def get_future_months(last_date: pd.Timestamp):
    """
    データ最終日の翌月から2カレンダー月分の期間を計算する。

    例: last_date = 2025/02/28
        FutureMonth1: 2025/03/01 ～ 2025/03/31
        FutureMonth2: 2025/04/01 ～ 2025/04/30
    """
    # Future Month 1
    fm1_month = last_date.month + 1 if last_date.month < 12 else 1
    fm1_year  = last_date.year if last_date.month < 12 else last_date.year + 1
    fm1_start = pd.Timestamp(fm1_year, fm1_month, 1)
    fm1_end   = pd.Timestamp(fm1_year, fm1_month,
                             calendar.monthrange(fm1_year, fm1_month)[1])

    # Future Month 2
    fm2_month = fm1_month + 1 if fm1_month < 12 else 1
    fm2_year  = fm1_year if fm1_month < 12 else fm1_year + 1
    fm2_start = pd.Timestamp(fm2_year, fm2_month, 1)
    fm2_end   = pd.Timestamp(fm2_year, fm2_month,
                             calendar.monthrange(fm2_year, fm2_month)[1])

    return fm1_start, fm1_end, fm2_start, fm2_end


def _calendar_row(d: pd.Timestamp) -> dict:
    """1日分のカレンダー特徴量を辞書で返す"""
    row = {
        "dow":           d.dayofweek,
        "month":         d.month,
        "day":           d.day,
        "dayofyear":     d.dayofyear,
        "week":          int(d.isocalendar()[1]),
        "quarter":       d.quarter,
        "is_weekend":    int(d.dayofweek >= 5),
        "is_month_start": int(d.day <= 3),
        "is_month_end":  int(d.day >= 28),
        "sin_dow":       np.sin(2 * np.pi * d.dayofweek / 7),
        "cos_dow":       np.cos(2 * np.pi * d.dayofweek / 7),
        "sin_month":     np.sin(2 * np.pi * d.month / 12),
        "cos_month":     np.cos(2 * np.pi * d.month / 12),
        "sin_doy":       np.sin(2 * np.pi * d.dayofyear / 365),
        "cos_doy":       np.cos(2 * np.pi * d.dayofyear / 365),
    }
    try:
        import jpholiday
        row["is_holiday"]     = int(jpholiday.is_holiday(d))
        row["is_golden_week"] = int((d.month == 4 and d.day >= 29) or
                                    (d.month == 5 and d.day <= 5))
        row["is_obon"]        = int(d.month == 8 and 13 <= d.day <= 15)
        row["is_year_end_ny"] = int((d.month == 12 and d.day >= 28) or
                                    (d.month == 1 and d.day <= 4))
    except ImportError:
        pass
    return row


def forecast_future(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    tsfresh_train_cols: list,
    window_size: int,
    fast: bool,
    train_median: pd.Series,
) -> tuple:
    """
    全データ（テスト含む）を使い、翌月・翌々月を予測する。

    ── アルゴリズム ──
    [STEP A] 未来全日付に対して tsfresh 特徴量を一括抽出
             - 窓 [t-W, t-1] を実データ + 直前予測値で構成
             - 実データが不足する部分は直近の実データで埋める
             ※ この段階では第1パス（粗い予測値で窓を補完）
    [STEP B] ラグ特徴量 + カレンダー特徴量を逐次計算し本予測
             - ラグ参照先が未来日付の場合は前STEP A予測値を使用

    Parameters
    ----------
    model             : 学習済みLGBMRegressor
    df                : 全データ（ds, y）
    feature_cols      : 学習時の特徴量リスト
    tsfresh_train_cols: 学習時に生成されたtsfresh列名（列整合のため）
    window_size       : tsfresh窓サイズ（日数）
    fast              : True=MinimalFCParameters
    train_median      : 学習データの列ごと中央値（欠損補完用）
    """
    last_date = df["ds"].max()
    fm1_start, fm1_end, fm2_start, fm2_end = get_future_months(last_date)
    future_dates = pd.date_range(fm1_start, fm2_end, freq="D")
    n_future = len(future_dates)

    logger.info(f"\n{'='*55}")
    logger.info(f"  未来予測（全データ使用）")
    logger.info(f"  FutureMonth1: {fm1_start.date()} ～ {fm1_end.date()} "
                f"({calendar.monthrange(fm1_year := fm1_start.year, fm1_start.month)[1]}日)")
    logger.info(f"  FutureMonth2: {fm2_start.date()} ～ {fm2_end.date()} "
                f"({calendar.monthrange(fm2_start.year, fm2_start.month)[1]}日)")
    logger.info(f"{'='*55}")

    # ── STEP A: tsfresh 特徴量を一括抽出 ─────────────────────────────────
    # 全データの y を Series に展開（実データのみ）
    history_actual = df.set_index("ds")["y"].copy()
    last_actual_val = float(history_actual.iloc[-1])

    logger.info("  [STEP A] tsfresh特徴量を未来全日付で一括抽出中...")
    records_future = []
    for pred_date in future_dates:
        w_start = pred_date - pd.Timedelta(days=window_size)
        w_end   = pred_date - pd.Timedelta(days=1)

        # 実データの範囲と交差する部分を取得
        avail = history_actual[
            (history_actual.index >= w_start) &
            (history_actual.index <= w_end)
        ]
        n_avail = len(avail)

        if n_avail >= window_size:
            window_vals = avail.values[-window_size:]
        elif n_avail > 0:
            # 実データが足りない部分は最初の実値でパディング
            pad = np.full(window_size - n_avail, float(avail.values[0]))
            window_vals = np.concatenate([pad, avail.values])
        else:
            # 実データが全くない（窓がデータ範囲外）は最終実値で埋める
            window_vals = np.full(window_size, last_actual_val)

        for t, v in enumerate(window_vals):
            records_future.append({"id": pred_date, "time": t, "value": v})

    df_rolled_future = pd.DataFrame(records_future)
    fc_params = MinimalFCParameters() if fast else EfficientFCParameters()
    tsfresh_future_raw = extract_features(
        df_rolled_future,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_params,
        n_jobs=1,
        show_warnings=False,
        disable_progressbar=False,
    )
    impute(tsfresh_future_raw)
    tsfresh_future_raw.index = pd.to_datetime(tsfresh_future_raw.index)
    logger.info(f"  [STEP A] 完了: {tsfresh_future_raw.shape[1]} 特徴量")

    # ── STEP B: 逐次予測（再帰的ラグ特徴量） ─────────────────────────────
    logger.info("  [STEP B] 逐次予測（再帰的ラグ補完）中...")
    # 履歴 = 実データ + 後から追記する予測値
    history_y = history_actual.copy()

    predictions_future = []

    for pred_date in future_dates:
        row = _calendar_row(pred_date)

        # ラグ特徴量（実データ or 既に予測した値を参照）
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            lag_date = pred_date - pd.Timedelta(days=lag)
            if lag_date in history_y.index:
                row[f"lag_{lag}"] = float(history_y[lag_date])
            else:
                # 参照先がまだ予測されていない（通常は起きないが安全策）
                row[f"lag_{lag}"] = float(history_y.iloc[-1])

        # tsfresh 特徴量を取得（STEP Aで抽出済み）
        if pred_date in tsfresh_future_raw.index:
            for col in tsfresh_train_cols:
                if col in tsfresh_future_raw.columns:
                    row[col] = float(tsfresh_future_raw.loc[pred_date, col])
                else:
                    row[col] = float(train_median.get(col, 0.0))

        # 特徴量ベクトルを学習時の列に揃える
        X_future = pd.DataFrame([row])
        for col in feature_cols:
            if col not in X_future.columns:
                X_future[col] = float(train_median.get(col, 0.0))
        X_future = X_future[feature_cols].fillna(0.0)

        y_pred = float(model.predict(X_future)[0])
        y_pred = max(0.0, y_pred)  # 非負制約

        period = "FutureMonth1" if pred_date <= fm1_end else "FutureMonth2"
        predictions_future.append({
            "ds":     pred_date,
            "y_pred": round(y_pred, 2),
            "period": period,
        })

        # 履歴に予測値を追記（次のラグ参照で使う）
        history_y[pred_date] = y_pred

    df_future = pd.DataFrame(predictions_future)
    logger.info(f"  [STEP B] 完了: {len(df_future)} 日分の予測")

    return df_future, fm1_start, fm1_end, fm2_start, fm2_end


def plot_future_forecast(
    df: pd.DataFrame,
    df_future: pd.DataFrame,
    df_test_result: pd.DataFrame,
    fm1_start, fm1_end, fm2_start, fm2_end,
    output_dir: str,
):
    """
    未来予測グラフを描画・保存する。

    構成:
    - (上) 全期間サマリー: 実績（全データ）+ テスト予測 + 未来予測
    - (下) 直近3ヶ月ズーム: テスト期間 + 未来予測を拡大表示
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))

    # ── (上) 全期間 ────────────────────────────────────────────────────────
    ax = axes[0]

    # 全実績（訓練+テスト）
    ax.plot(df["ds"], df["y"],
            color="#475569", linewidth=1.2, alpha=0.6, label="実績（全データ）")

    # テスト期間の予測（後ろに重ねる）
    if df_test_result is not None and len(df_test_result) > 0:
        ax.plot(df_test_result["ds"], df_test_result["y_pred"],
                color="#F97316", linewidth=1.8, linestyle="--",
                label=f"テスト予測 ({df_test_result['ds'].min().strftime('%Y/%m')}～"
                      f"{df_test_result['ds'].max().strftime('%Y/%m')})",
                alpha=0.85)

    # 未来予測（Month1・Month2 それぞれ色分け）
    fm1_data = df_future[df_future["period"] == "FutureMonth1"]
    fm2_data = df_future[df_future["period"] == "FutureMonth2"]

    ax.plot(fm1_data["ds"], fm1_data["y_pred"],
            color="#2563EB", linewidth=2.5, marker="o", markersize=3,
            label=f"未来予測Month1 ({fm1_start.strftime('%Y/%m')})")
    ax.plot(fm2_data["ds"], fm2_data["y_pred"],
            color="#7C3AED", linewidth=2.5, marker="o", markersize=3,
            label=f"未来予測Month2 ({fm2_start.strftime('%Y/%m')})")

    # 境界線
    ax.axvline(fm1_start, color="#2563EB", linestyle=":", linewidth=1.5,
               label=f"予測開始 ({fm1_start.date()})")
    ax.axvline(fm2_start, color="#7C3AED", linestyle=":", linewidth=1.5)

    ax.set_title("予測結果サマリー（実績 + テスト予測 + 未来予測）",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("日付")
    ax.set_ylabel("呼量 (y)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)

    # ── (下) 直近3ヶ月ズーム ─────────────────────────────────────────────
    ax2 = axes[1]

    # ズーム範囲: テスト開始の1ヶ月前 ～ 未来予測終了
    zoom_start = fm1_start - pd.DateOffset(months=3)
    zoom_df    = df[(df["ds"] >= zoom_start)]

    ax2.plot(zoom_df["ds"], zoom_df["y"],
             color="#1F2937", linewidth=2, marker="o", markersize=3,
             label="実績")

    if df_test_result is not None and len(df_test_result) > 0:
        ax2.plot(df_test_result["ds"], df_test_result["y_pred"],
                 color="#F97316", linewidth=2, linestyle="--",
                 marker="^", markersize=4,
                 label="テスト予測")

    ax2.plot(fm1_data["ds"], fm1_data["y_pred"],
             color="#2563EB", linewidth=2.5, linestyle="-",
             marker="s", markersize=5,
             label=f"未来Month1 ({fm1_start.strftime('%Y/%m')})")
    ax2.plot(fm2_data["ds"], fm2_data["y_pred"],
             color="#7C3AED", linewidth=2.5, linestyle="-",
             marker="D", markersize=5,
             label=f"未来Month2 ({fm2_start.strftime('%Y/%m')})")

    # 月背景色
    ax2.axvspan(fm1_start, fm1_end, alpha=0.08, color="#2563EB",
                label=f"FutureMonth1")
    ax2.axvspan(fm2_start, fm2_end, alpha=0.08, color="#7C3AED",
                label=f"FutureMonth2")

    # 境界線（テストとの境）
    ax2.axvline(fm1_start, color="#2563EB", linestyle=":", linewidth=1.8)

    ax2.set_title(
        f"直近ズーム（テスト＋未来予測 | "
        f"FutureM1: {fm1_start.date()}～{fm1_end.date()}, "
        f"FutureM2: {fm2_start.date()}～{fm2_end.date()}）",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("日付")
    ax2.set_ylabel("呼量 (y)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "09_future_forecast.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 未来予測グラフ保存: {save_path}")


def plot_future_monthly_bar(
    df_future: pd.DataFrame,
    fm1_start, fm1_end, fm2_start, fm2_end,
    output_dir: str,
):
    """
    未来2ヶ月の月別集計グラフ（合計・平均・最大・最小）
    """
    fm1_data = df_future[df_future["period"] == "FutureMonth1"]
    fm2_data = df_future[df_future["period"] == "FutureMonth2"]

    stats = {}
    for label, data, s, e in [
        (f"FutureMonth1\n({fm1_start.strftime('%Y/%m')})", fm1_data, fm1_start, fm1_end),
        (f"FutureMonth2\n({fm2_start.strftime('%Y/%m')})", fm2_data, fm2_start, fm2_end),
    ]:
        stats[label] = {
            "合計":  round(data["y_pred"].sum(), 1),
            "平均":  round(data["y_pred"].mean(), 1),
            "最大":  round(data["y_pred"].max(), 1),
            "最小":  round(data["y_pred"].min(), 1),
            "中央値": round(data["y_pred"].median(), 1),
            "日数":  len(data),
        }

    n_stats = 5  # 合計,平均,最大,最小,中央値
    fig, axes = plt.subplots(1, n_stats, figsize=(16, 5))
    stat_keys = ["合計", "平均", "最大", "最小", "中央値"]
    colors    = ["#2563EB", "#7C3AED"]

    for ax, stat in zip(axes, stat_keys):
        labels = list(stats.keys())
        vals   = [stats[l][stat] for l in labels]
        bars = ax.bar(labels, vals, color=colors[:len(labels)],
                      edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    f"{val:,.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(stat, fontsize=12, fontweight="bold")
        ax.set_ylabel(stat)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_facecolor("#F9FAFB")
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=9)

    fig.suptitle("未来2ヶ月 月別集計（予測値）", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "10_future_monthly_stats.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 未来月別集計グラフ保存: {save_path}")

    return stats


def save_future_report(
    df_future: pd.DataFrame,
    future_stats: dict,
    fm1_start, fm1_end, fm2_start, fm2_end,
    output_dir: str,
):
    """未来予測結果をCSV + テキストで保存"""
    # 予測値CSV
    csv_path = os.path.join(output_dir, "future_predictions.csv")
    df_future.to_csv(csv_path, index=False)
    logger.info(f"  📄 未来予測CSV保存: {csv_path}")

    # サマリーテキスト
    lines = [
        "=" * 55,
        "  未来予測 サマリー",
        "=" * 55,
        "",
        f"FutureMonth1: {fm1_start.date()} ～ {fm1_end.date()}",
        f"FutureMonth2: {fm2_start.date()} ～ {fm2_end.date()}",
        "",
    ]
    for period, st in future_stats.items():
        lines.append(f"▼ {period.strip()}")
        for k, v in st.items():
            lines.append(f"  {k:<8}: {v:,.1f}" if isinstance(v, float)
                         else f"  {k:<8}: {v}")
        lines.append("")

    txt_path = os.path.join(output_dir, "future_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  📄 未来レポート保存: {txt_path}")
    logger.info("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# 8. 結果レポートの保存
# ──────────────────────────────────────────────────────────────────────────────
def save_report(df_test: pd.DataFrame, metrics_dict: dict,
                best_params: dict, split_dates: dict,
                n_features: int, output_dir: str):
    """結果サマリーをテキスト + CSV で保存"""
    # 予測結果CSV
    df_test.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # サマリーレポート
    report_lines = [
        "=" * 60,
        "  tsfresh + LightGBM 時系列予測 結果レポート",
        "=" * 60,
        "",
        f"【データ期間】",
        f"  訓練期間     : ～ {split_dates['train_end'].date()}",
        f"  テスト全体   : {split_dates['test_start'].date()} ～ {split_dates['test_end'].date()}",
        f"  Month1 (1月) : {split_dates['m1_start'].date()} ～ {split_dates['m1_end'].date()}",
        f"  Month2 (2月) : {split_dates['m2_start'].date()} ～ {split_dates['m2_end'].date()}",
        "",
        f"【特徴量数】: {n_features}",
        "",
        "【評価指標】",
    ]

    for period, mets in metrics_dict.items():
        report_lines.append(f"\n  ▼ {period}")
        for k, v in mets.items():
            report_lines.append(f"    {k:<15}: {v}")

    report_lines += [
        "",
        "【Optuna最良パラメータ】",
    ]
    for k, v in best_params.items():
        if k not in ("objective", "metric", "verbosity"):
            report_lines.append(f"  {k:<25}: {v}")

    report_text = "\n".join(report_lines)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"\n{'='*55}")
    logger.info(report_text)
    logger.info(f"{'='*55}")
    logger.info(f"  📄 レポート保存: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="tsfresh + LightGBM 時系列予測"
    )
    parser.add_argument("--csv",     type=str, required=True,
                        help="入力CSVファイル（ds, y列）")
    parser.add_argument("--window",  type=int, default=DEFAULT_WINDOW,
                        help=f"tsfresh ローリング窓サイズ（日数, デフォルト: {DEFAULT_WINDOW}）")
    parser.add_argument("--trials",  type=int, default=DEFAULT_TRIALS,
                        help=f"Optuna 試行回数（デフォルト: {DEFAULT_TRIALS}）")
    parser.add_argument("--cv_folds",type=int, default=DEFAULT_CV_FOLDS,
                        help=f"TimeSeriesSplit フォールド数（デフォルト: {DEFAULT_CV_FOLDS}）")
    parser.add_argument("--fast",    action="store_true",
                        help="tsfresh MinimalFCParameters を使用（高速化）")
    parser.add_argument("--top_n",   type=int, default=30,
                        help="Feature Importance / SHAP の表示上位数（デフォルト: 30）")
    parser.add_argument("--out",     type=str, default=OUTPUT_DIR,
                        help="出力フォルダ名")
    args = parser.parse_args()

    # 出力フォルダ作成
    os.makedirs(args.out, exist_ok=True)
    logger.info(f"出力フォルダ: {args.out}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: データ読み込み
    # ────────────────────────────────────────────────────────────────────────
    df = load_data(args.csv)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: テスト期間をカレンダー月で厳密に計算
    # ────────────────────────────────────────────────────────────────────────
    split_dates = get_test_split_dates(df)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: tsfresh ローリング特徴量生成
    # ────────────────────────────────────────────────────────────────────────
    tsfresh_features = build_tsfresh_features(df, args.window, fast=args.fast)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: カレンダー + ラグ特徴量の追加
    # ────────────────────────────────────────────────────────────────────────
    logger.info("カレンダー・ラグ特徴量を追加中...")
    df_feat = df.copy()
    df_feat = add_calendar_features(df_feat)
    df_feat = add_lag_features(df_feat, lags=[1, 2, 3, 7, 14, 21, 28])
    df_feat = df_feat.set_index("ds")

    # tsfresh特徴量と結合（共通のdateインデックスで内部結合）
    df_combined = df_feat.join(tsfresh_features, how="inner")

    logger.info(f"  結合後: {len(df_combined)} 行 × {df_combined.shape[1]} 列")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: 学習/テスト分割
    # ────────────────────────────────────────────────────────────────────────
    # y列を分離
    target_col = "y"
    # カレンダー列以外の不要列を除去
    drop_cols = [target_col]
    feature_cols = [c for c in df_combined.columns if c not in drop_cols]

    # 訓練データ
    train_mask = df_combined.index <= split_dates["train_end"]
    test_mask  = (df_combined.index >= split_dates["test_start"]) & \
                 (df_combined.index <= split_dates["test_end"])

    X_all = df_combined[feature_cols].copy()
    y_all = df_combined[target_col].copy()

    # ラグ特徴量のNaNを除去（訓練のみ）
    X_train_full = X_all[train_mask].dropna()
    y_train_full = y_all[X_train_full.index]

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    logger.info(f"\n  訓練サンプル数 : {len(X_train_full)}")
    logger.info(f"  テストサンプル数: {len(X_test)}")
    logger.info(f"  特徴量数        : {len(feature_cols)}")

    # 最後の20%を検証データとして使用（Optuna用）
    val_cutoff = int(len(X_train_full) * 0.8)
    X_train_opt = X_train_full.iloc[:val_cutoff]
    y_train_opt = y_train_full.iloc[:val_cutoff]

    # テストデータのNaN処理（将来ラグ等で生じるNaNを中央値で補完）
    X_test_filled = X_test.fillna(X_train_full.median())

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: Optuna ハイパーパラメータ最適化
    # ────────────────────────────────────────────────────────────────────────
    best_params, study = optuna_optimize(
        X_train_opt, y_train_opt,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: 最終モデル訓練
    # ────────────────────────────────────────────────────────────────────────
    # 検証セット（最後の15%）
    final_val_cutoff = int(len(X_train_full) * 0.85)
    X_tr_final = X_train_full.iloc[:final_val_cutoff]
    y_tr_final = y_train_full.iloc[:final_val_cutoff]
    X_val_final = X_train_full.iloc[final_val_cutoff:]
    y_val_final = y_train_full.iloc[final_val_cutoff:]

    final_model = train_final_model(X_tr_final, y_tr_final,
                                    X_val_final, y_val_final, best_params)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 8: 予測と評価
    # ────────────────────────────────────────────────────────────────────────
    logger.info("\nテスト期間の予測中...")
    y_pred = final_model.predict(X_test_filled)

    # テスト結果DataFrame
    df_test_result = pd.DataFrame({
        "ds":       y_test.index,
        "y":        y_test.values,
        "y_pred":   y_pred,
        "residual": y_test.values - y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
        "pct_error": np.abs((y_test.values - y_pred) / (np.abs(y_test.values) + 1e-8)) * 100,
    })

    # ── 月別にperiodラベルを付与 ──────────────────────────────────────────
    def assign_period(ds):
        if split_dates["m1_start"] <= ds <= split_dates["m1_end"]:
            return "Month1"
        elif split_dates["m2_start"] <= ds <= split_dates["m2_end"]:
            return "Month2"
        else:
            return "Other"

    df_test_result["period"] = df_test_result["ds"].apply(assign_period)

    # ── 評価指標計算 ──────────────────────────────────────────────────────
    metrics_dict = {}

    # テスト全体
    metrics_dict["Test_All"] = compute_metrics(
        df_test_result["y"], df_test_result["y_pred"],
        label=f"テスト全体 ({split_dates['test_start'].date()} ～ {split_dates['test_end'].date()})"
    )

    # Month1
    m1_data = df_test_result[df_test_result["period"] == "Month1"]
    if len(m1_data) > 0:
        metrics_dict[f"Month1 ({split_dates['m1_start'].strftime('%Y/%m')})"] = compute_metrics(
            m1_data["y"], m1_data["y_pred"],
            label=f"Month1: {split_dates['m1_start'].date()} ～ {split_dates['m1_end'].date()} ({len(m1_data)}日)"
        )
    else:
        logger.warning("Month1のデータが存在しません")

    # Month2
    m2_data = df_test_result[df_test_result["period"] == "Month2"]
    if len(m2_data) > 0:
        metrics_dict[f"Month2 ({split_dates['m2_start'].strftime('%Y/%m')})"] = compute_metrics(
            m2_data["y"], m2_data["y_pred"],
            label=f"Month2: {split_dates['m2_start'].date()} ～ {split_dates['m2_end'].date()} ({len(m2_data)}日)"
        )
    else:
        logger.warning("Month2のデータが存在しません")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 9: 可視化
    # ────────────────────────────────────────────────────────────────────────
    logger.info("\n可視化グラフを生成中...")

    # 訓練データ用のdf（可視化用）
    df_train_vis = df[df["ds"] <= split_dates["train_end"]].copy()

    plot_forecast(df_test_result, df_train_vis, split_dates, args.out)
    plot_metrics_table(metrics_dict, args.out)
    plot_monthly_metrics(metrics_dict, args.out)
    plot_feature_importance(final_model, feature_cols, args.out, top_n=args.top_n)
    plot_shap(final_model, X_test_filled, args.out, top_n=args.top_n)
    plot_residuals(df_test_result, args.out)
    plot_optuna_results(study, args.out)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 10: レポート保存
    # ────────────────────────────────────────────────────────────────────────
    save_report(
        df_test_result, metrics_dict, best_params,
        split_dates, len(feature_cols), args.out
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 11: 未来予測（全データ → 翌月・翌々月）
    # ────────────────────────────────────────────────────────────────────────
    logger.info("\n未来予測（全データ使用）を開始...")

    # 全データでモデルを再訓練（テストデータも学習に含める）
    logger.info("  全データでモデルを再訓練中...")
    X_all_clean = X_all.dropna()
    y_all_clean = y_all[X_all_clean.index]

    # 検証用に最後の10%を使用
    final_all_cutoff = int(len(X_all_clean) * 0.90)
    X_tr_all  = X_all_clean.iloc[:final_all_cutoff]
    y_tr_all  = y_all_clean.iloc[:final_all_cutoff]
    X_val_all = X_all_clean.iloc[final_all_cutoff:]
    y_val_all = y_all_clean.iloc[final_all_cutoff:]

    model_all = train_final_model(
        X_tr_all, y_tr_all,
        X_val_all, y_val_all,
        best_params
    )

    # 学習データ中央値（欠損補完用）
    train_median = X_all_clean.median()

    # tsfresh列名を特定（feature_colsの中でtsfresh由来の列）
    lag_cal_cols = [
        "dow", "month", "day", "dayofyear", "week", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "sin_dow", "cos_dow", "sin_month", "cos_month", "sin_doy", "cos_doy",
        "is_holiday", "is_golden_week", "is_obon", "is_year_end_ny",
    ] + [f"lag_{l}" for l in [1, 2, 3, 7, 14, 21, 28]]
    tsfresh_train_cols = [c for c in feature_cols if c not in lag_cal_cols]

    # 未来予測実行
    df_future, fm1_start, fm1_end, fm2_start, fm2_end = forecast_future(
        model=model_all,
        df=df,
        feature_cols=feature_cols,
        tsfresh_train_cols=tsfresh_train_cols,
        window_size=args.window,
        fast=args.fast,
        train_median=train_median,
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 12: 未来予測の可視化・レポート
    # ────────────────────────────────────────────────────────────────────────
    logger.info("\n未来予測グラフを生成中...")
    plot_future_forecast(
        df=df,
        df_future=df_future,
        df_test_result=df_test_result,
        fm1_start=fm1_start, fm1_end=fm1_end,
        fm2_start=fm2_start, fm2_end=fm2_end,
        output_dir=args.out,
    )

    future_stats = plot_future_monthly_bar(
        df_future=df_future,
        fm1_start=fm1_start, fm1_end=fm1_end,
        fm2_start=fm2_start, fm2_end=fm2_end,
        output_dir=args.out,
    )

    save_future_report(
        df_future=df_future,
        future_stats=future_stats,
        fm1_start=fm1_start, fm1_end=fm1_end,
        fm2_start=fm2_start, fm2_end=fm2_end,
        output_dir=args.out,
    )

    logger.info("\n" + "=" * 55)
    logger.info("  ✅ 全処理完了！")
    logger.info(f"  出力フォルダ: {os.path.abspath(args.out)}")
    logger.info("  生成ファイル一覧:")
    for f in sorted(os.listdir(args.out)):
        fpath = os.path.join(args.out, f)
        size_kb = os.path.getsize(fpath) / 1024
        logger.info(f"    {f:<45} ({size_kb:.1f} KB)")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
