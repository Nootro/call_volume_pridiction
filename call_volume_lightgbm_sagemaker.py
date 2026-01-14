#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SageMaker 向けに日次呼量データからマルチステップ予測を行う LightGBM モデルです。

このスクリプトはベースラインの LightGBM 実装を拡張し、長期・中期の季節性を捉えるための
ラグ特徴量や移動平均、時系列交差検証、予測値を用いて将来を iteratively 予測する機能などを備えています。
`ds`（日付）列と `y`（呼量）列を含む CSV ファイルを読み込むことも、
ファイルが指定されていない場合は合成データを生成することもできます。
パラメータはコード内で設定され、コマンドライン引数は使用しません。
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "LightGBM が必要です。`pip install lightgbm` でインストールしてください。"
    ) from exc

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


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
    """日付と呼量から各種特徴量を生成します。"""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    # ラグ特徴量：日次、週次、月次、四半期、半年、年次
    lag_days = [1, 7, 14, 28, 30, 60, 90, 180, 365]
    for lag in lag_days:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # ローリング統計
    windows = [7, 14, 30, 60, 90, 180]
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w).std()
    # 指数移動平均
    for span in [7, 30, 90]:
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
    # 月境界および平日のフラグ
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    # 週末後最初の営業日を示すフラグ（間隔が 1 日より長い場合）
    df["is_first_after_weekend"] = (df["ds"].diff().dt.days > 1).astype(int)
    # 月の最終平日を示すフラグ（月末かつ週末でない日）
    df["is_last_weekday"] = (
        df["ds"].dt.is_month_end & ~df["ds"].dt.dayofweek.isin([5, 6])
    ).astype(int)
    # 周期エンコーディング
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    # 祝日フラグ
    holidays = generate_jp_holidays(df["ds"].min(), df["ds"].max())
    df["is_holiday"] = df["ds"].isin(holidays).astype(int)
    # 前祝日フラグ：祝日の前日
    if holidays:
        pre_holiday_set = {h - pd.Timedelta(days=1) for h in holidays}
        df["is_pre_holiday"] = df["ds"].isin(pre_holiday_set).astype(int)
    else:
        df["is_pre_holiday"] = 0
    # 傾向と相対変化の特徴量
    df["diff_1"] = df["y"] - df["lag_1"]
    df["diff_7"] = df["y"] - df["lag_7"]
    df["diff_30"] = df["y"] - df["lag_30"]
    for w in [30, 90]:
        roll_mean = df["y"].rolling(w).mean()
        df[f"rel_change_{w}"] = (df["y"] - roll_mean) / roll_mean
    # NaN を含む行を削除
    df = df.dropna().reset_index(drop=True)
    return df


# -- 時系列交差検証 --
def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[float], lgb.LGBMRegressor]:
    """時系列データに対して rolling-origin 交差検証を行い、最終モデルを訓練します。

    各フォールドの RMSE スコアと、全データで学習した最終モデルを返します。
    """
    # 特徴量とターゲットの準備
    X = df.drop(columns=["ds", "y"])
    y = df["y"]
    categorical_features = [
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "quarter",
        "year",
        "is_weekend",
        "is_holiday",
    ]
    # カテゴリ列を整数に変換
    for col in categorical_features:
        X[col] = X[col].astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores: List[float] = []
    # LightGBM のハイパーパラメータ
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": random_state,
    }
    # 各フォールドでモデルを訓練
    fold = 0
    for train_index, valid_index in tscv.split(X):
        fold += 1
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = lgb.LGBMRegressor(n_estimators=2000, **params)
        # コールバックを準備
        callbacks = []
        try:
            callbacks.append(lgb.early_stopping(stopping_rounds=100))
        except AttributeError:
            pass
        try:
            callbacks.append(lgb.log_evaluation(period=200))
        except AttributeError:
            pass
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            categorical_feature=categorical_features,
            callbacks=callbacks,
        )
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_scores.append(rmse)
        print(f"Fold {fold}/{n_splits} RMSE: {rmse:.4f}")
    # 全データで最終モデルを訓練
    final_model = lgb.LGBMRegressor(n_estimators=2000, **params)
    callbacks = []
    try:
        callbacks.append(lgb.early_stopping(stopping_rounds=100))
    except AttributeError:
        pass
    try:
        callbacks.append(lgb.log_evaluation(period=200))
    except AttributeError:
        pass
    final_model.fit(
        X,
        y,
        eval_set=[(X, y)],
        eval_metric="rmse",
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )
    return rmse_scores, final_model


# -- 将来予測 --
def forecast_future(
    df: pd.DataFrame,
    model: lgb.LGBMRegressor,
    horizon: int,
) -> pd.DataFrame:
    """学習済みモデルを用いて将来の予測を生成します。"""
    future_predictions: List[dict] = []
    # 1 日ずつ予測して df を順次拡張する
    current_df = df.copy()
    for _ in range(horizon):
        # 最終日付から次の日付を計算
        last_date = current_df["ds"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        # 特徴量を生成するため、ds と y が NaN の行を追加
        new_row = pd.DataFrame({"ds": [next_date], "y": [np.nan]})
        temp_df = pd.concat([current_df, new_row], ignore_index=True)
        # 特徴量を計算し、新しい行を取得
        temp_features = create_features(temp_df)[-1:]
        X_new = temp_features.drop(columns=["ds", "y"])
        # 予測
        y_pred = model.predict(X_new, num_iteration=model.best_iteration_)[0]
        future_predictions.append({"ds": next_date, "y_pred": y_pred})
        # df を更新
        current_df = pd.concat(
            [current_df, pd.DataFrame({"ds": [next_date], "y": [y_pred]})],
            ignore_index=True,
        )
    return pd.DataFrame(future_predictions)


# -- 合成データ生成（ベースラインと同じ） --
def generate_synthetic_data(n_days: int = 3 * 365) -> pd.DataFrame:
    """単純なトレンドと季節性を持つ合成データを生成します。"""
    date_range = pd.date_range(
        start=pd.Timestamp.today() - pd.Timedelta(days=n_days), periods=n_days, freq="D"
    )
    trend = np.linspace(100, 200, n_days)
    weekly = 20 * np.sin(2 * np.pi * date_range.dayofweek / 7.0)
    annual = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.0)
    noise = np.random.normal(scale=5, size=n_days)
    y = trend + weekly + annual + noise
    return pd.DataFrame({"ds": date_range, "y": y})


def main() -> None:
    """デフォルト設定でモデルを学習し、将来予測を実行します。"""
    # 設定値（必要に応じて変更してください）
    data_file: str | None = None  # CSV ファイルへのパスを指定するとデータを読み込みます。None で合成データ生成。
    cv_splits: int = 5  # 交差検証の分割数
    horizon: int = 60  # 予測したい日数

    # データ読み込みまたは生成
    if data_file:
        df_raw = pd.read_csv(data_file)
        if "ds" not in df_raw.columns or "y" not in df_raw.columns:
            raise ValueError("入力 CSV には 'ds' と 'y' の列が必要です。")
    else:
        print("データファイルが指定されていないため、合成データを生成します。")
        df_raw = generate_synthetic_data(n_days=3 * 365)

    # 特徴量エンジニアリング
    df_features = create_features(df_raw)

    # 時系列交差検証
    rmse_scores, final_model = time_series_cv(df_features, n_splits=cv_splits)
    print(f"平均 CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    # 将来予測
    future_df = forecast_future(df_features, final_model, horizon)
    print(f"\nマルチステップ予測 (次の {horizon} 日):")
    print(future_df.head())


if __name__ == "__main__":
    main()