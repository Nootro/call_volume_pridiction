#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SageMaker 向けに TabPFN を使用した日次呼量データのマルチステップ予測モデルです。

TabPFN (Tabular Prior-Fitted Networks) は事前学習済みのTransformerベースモデルで、
小〜中規模データセット（最大10,000サンプル）で優れた性能を発揮します。
学習が不要で推論のみで動作するため、非常に高速です。

注意: TabPFN は最大 10,000 サンプル、1,000 特徴量までの制限があります。
大規模データの場合は、最新データのサブセットを使用します。
"""

import logging
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from tabpfn import TabPFNRegressor
except ImportError as exc:
    raise SystemExit(
        "tabpfn が必要です。`pip install tabpfn` でインストールしてください。\n"
        "注: PyTorch も必要です。`pip install torch` でインストールしてください。"
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


# -- 特徴量エンジニアリング（TabPFN 向けに最適化） --
def create_features_tabpfn(df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """日付と呼量から重要な特徴量のみを生成します。
    
    TabPFN は特徴量数の制限があるため、最も重要な特徴量に絞ります。
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    # 重要なラグ特徴量のみ
    lag_days = [1, 7, 14, 28, 30, 60, 90]
    for lag in lag_days:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    
    # ローリング統計（重要なウィンドウのみ）
    windows = [7, 14, 30]
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w, min_periods=1).std()
    
    # 指数移動平均
    for span in [7, 30]:
        df[f"ewm_{span}"] = df["y"].ewm(span=span, adjust=False).mean()
    
    # カレンダー属性
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["ds"].dt.month
    df["day_of_month"] = df["ds"].dt.day
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["ds"].dt.quarter
    
    # 月境界フラグ
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    
    # 周期エンコーディング（最も重要なもののみ）
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    # 祝日フラグ
    holidays = generate_jp_holidays(df["ds"].min(), df["ds"].max())
    df["is_holiday"] = df["ds"].isin(holidays).astype(int)
    
    # 前祝日フラグ
    if holidays:
        pre_holiday_set = {h - pd.Timedelta(days=1) for h in holidays}
        df["is_pre_holiday"] = df["ds"].isin(pre_holiday_set).astype(int)
    else:
        df["is_pre_holiday"] = 0
    
    # 差分特徴量
    df["diff_1"] = df["y"] - df["lag_1"]
    df["diff_7"] = df["y"] - df["lag_7"]
    
    # 相対変化率
    for w in [7, 30]:
        roll_mean = df["y"].rolling(w, min_periods=1).mean()
        df[f"rel_change_{w}"] = (df["y"] - roll_mean) / (roll_mean + 1e-8)
    
    # NaN を前方埋めと後方埋めで処理
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.fillna(0)
    
    return df


# -- データサンプリング（TabPFN の制限対応） --
def sample_data_for_tabpfn(
    df: pd.DataFrame, 
    max_samples: int = 8000
) -> pd.DataFrame:
    """TabPFN の制限（10,000サンプル）に対応するため、最新データをサンプリングします。
    
    時系列の連続性を保つため、最新のデータから取得します。
    """
    if len(df) <= max_samples:
        return df
    
    print(f"データサイズが {len(df)} サンプルのため、最新 {max_samples} サンプルを使用します。")
    return df.iloc[-max_samples:].reset_index(drop=True)


# -- 時系列交差検証 with TabPFN --
def time_series_cv_tabpfn(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[float], List[TabPFNRegressor]]:
    """時系列データに対して rolling-origin 交差検証を行います。

    TabPFN は学習が不要（推論のみ）なので、各フォールドで新しいモデルを作成します。
    各フォールドのモデルを保存し、アンサンブル予測に使用します。
    """
    # TabPFN のサンプル数制限に対応
    df = sample_data_for_tabpfn(df, max_samples=8000)
    
    # 特徴量とターゲットの準備
    X = df.drop(columns=["ds", "y"]).values
    y = df["y"].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores: List[float] = []
    models: List[TabPFNRegressor] = []
    
    # TabPFN のパラメータ
    tabpfn_params = {
        "device": "cpu",              # 'cuda' または 'cpu'
        "N_ensemble_configurations": 16,  # アンサンブル数（精度向上）
    }
    
    # 各フォールドでモデルを評価
    fold = 0
    for train_index, valid_index in tscv.split(X):
        fold += 1
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        # TabPFN モデルを作成（学習は不要、fit で推論準備のみ）
        model = TabPFNRegressor(**tabpfn_params)
        
        # fit は訓練データを保存するだけ（実際の学習は行わない）
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_scores.append(rmse)
        models.append(model)
        
        print(f"Fold {fold}/{n_splits} RMSE: {rmse:.4f}")
    
    return rmse_scores, models


# -- アンサンブル予測 --
def ensemble_predict(
    models: List[TabPFNRegressor],
    X: np.ndarray
) -> np.ndarray:
    """複数の TabPFN モデルのアンサンブル予測を行います。"""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # 平均を取る
    return np.mean(predictions, axis=0)


# -- 将来予測 --
def forecast_future(
    df: pd.DataFrame,
    models: List[TabPFNRegressor],
    horizon: int,
) -> pd.DataFrame:
    """学習済み TabPFN モデル（アンサンブル）を用いて将来の予測を生成します。"""
    future_predictions: List[dict] = []
    current_df = df.copy()
    
    for step in range(horizon):
        # 最終日付から次の日付を計算
        last_date = current_df["ds"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # 特徴量を生成するため、ds と y が NaN の行を追加
        new_row = pd.DataFrame({"ds": [next_date], "y": [np.nan]})
        temp_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # 特徴量を計算し、新しい行を取得
        temp_features = create_features_tabpfn(temp_df)
        last_row = temp_features.iloc[-1:]
        
        X_new = last_row.drop(columns=["ds", "y"]).values
        
        # アンサンブル予測
        y_pred = ensemble_predict(models, X_new)[0]
        future_predictions.append({"ds": next_date, "y_pred": y_pred})
        
        # df を更新（予測値を実際の値として使用）
        current_df = pd.concat(
            [current_df, pd.DataFrame({"ds": [next_date], "y": [y_pred]})],
            ignore_index=True,
        )
        
        if (step + 1) % 10 == 0:
            print(f"予測進捗: {step + 1}/{horizon} 日")
    
    return pd.DataFrame(future_predictions)


# -- 合成データ生成 --
def generate_synthetic_data(n_days: int = 2 * 365) -> pd.DataFrame:
    """複雑なトレンドと季節性を持つ合成データを生成します。
    
    TabPFN の制限を考慮して、デフォルトで2年分のデータを生成。
    """
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
    print("TabPFN 呼量予測モデル")
    print("=" * 60)
    print("\n注意: TabPFN は最大 10,000 サンプルまで対応しています。")
    print("大規模データの場合は最新データのサブセットを使用します。\n")
    
    # 設定値（必要に応じて変更してください）
    data_file: Optional[str] = None  # CSV ファイルへのパスを指定
    cv_splits: int = 5                # 交差検証の分割数
    horizon: int = 60                 # 予測したい日数
    random_state: int = 42

    # データ読み込みまたは生成
    if data_file:
        df_raw = pd.read_csv(data_file)
        if "ds" not in df_raw.columns or "y" not in df_raw.columns:
            raise ValueError("入力 CSV には 'ds' と 'y' の列が必要です。")
    else:
        print("データファイルが指定されていないため、合成データを生成します。")
        df_raw = generate_synthetic_data(n_days=2 * 365)
    
    print(f"データサイズ: {len(df_raw)} 日分")

    # 特徴量エンジニアリング
    print("\n特徴量を生成中...")
    df_features = create_features_tabpfn(df_raw)
    print(f"生成された特徴量数: {len(df_features.columns) - 2} (ds, y を除く)")

    # 時系列交差検証
    print("\n時系列交差検証を実行中（TabPFN は推論のみで高速）...")
    rmse_scores, models = time_series_cv_tabpfn(
        df_features, 
        n_splits=cv_splits,
        random_state=random_state
    )
    print(f"\n平均 CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    
    # 各フォールドのスコアを表示
    print("\n各フォールドの RMSE:")
    for i, score in enumerate(rmse_scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    # 将来予測（アンサンブル）
    print(f"\n将来 {horizon} 日の予測を生成中（{len(models)} モデルのアンサンブル）...")
    future_df = forecast_future(df_features, models, horizon)
    print(f"\nマルチステップ予測 (次の {horizon} 日):")
    print(future_df.head(20))
    
    # 統計サマリー
    print(f"\n予測統計:")
    print(f"  平均: {future_df['y_pred'].mean():.2f}")
    print(f"  標準偏差: {future_df['y_pred'].std():.2f}")
    print(f"  最小値: {future_df['y_pred'].min():.2f}")
    print(f"  最大値: {future_df['y_pred'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("TabPFN の特徴:")
    print("  ✓ 事前学習済みモデルで学習不要")
    print("  ✓ 推論のみで非常に高速")
    print("  ✓ 小〜中規模データで優れた性能")
    print("  ✗ 最大 10,000 サンプル、1,000 特徴量の制限")
    print("=" * 60)


if __name__ == "__main__":
    main()
