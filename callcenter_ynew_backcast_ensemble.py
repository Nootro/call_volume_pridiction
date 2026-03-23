from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EPS = 1e-9
RANDOM_STATE = 42


@dataclass
class BackcastConfig:
    date_col: str = "ds"
    old_col: str = "y_old"
    new_col: str = "y_new"
    n_splits: int = 3
    test_size: int = 21
    gap: int = 7
    tsfresh_windows: Tuple[int, ...] = (28, 56)
    top_k_features: int = 140
    use_prophet: bool = True
    lgbm_n_estimators: int = 1200
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 31
    lgbm_subsample: float = 0.85
    lgbm_colsample_bytree: float = 0.85
    lgbm_reg_alpha: float = 0.2
    lgbm_reg_lambda: float = 1.0
    lgbm_min_child_samples: int = 12
    ratio_clip_quantiles: Tuple[float, float] = (0.01, 0.99)
    monthly_fourier_order: int = 5
    save_model_artifacts: bool = False


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + EPS))


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPS)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_frame(y_true, y_pred, model_name: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": model_name,
            "MAE": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
            "RMSE": rmse(y_true, y_pred),
            "sMAPE": float(smape(y_true, y_pred)),
            "WAPE": float(wape(y_true, y_pred)),
        }
    ])


def _safe_import_dependencies():
    from scipy.optimize import nnls
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.isotonic import IsotonicRegression
    from lightgbm import LGBMRegressor

    try:
        from prophet import Prophet
        prophet_ok = True
    except Exception:
        Prophet = None
        prophet_ok = False

    try:
        import jpholiday
    except Exception as e:
        raise ImportError("jpholiday が見つかりません。pip install jpholiday を実行してください。") from e

    try:
        from tsfresh.feature_extraction import EfficientFCParameters, extract_features
        from tsfresh.utilities.dataframe_functions import impute
    except Exception as e:
        raise ImportError("tsfresh が見つかりません。pip install tsfresh を実行してください。") from e

    return {
        "nnls": nnls,
        "TimeSeriesSplit": TimeSeriesSplit,
        "IsotonicRegression": IsotonicRegression,
        "LGBMRegressor": LGBMRegressor,
        "Prophet": Prophet,
        "prophet_ok": prophet_ok,
        "jpholiday": jpholiday,
        "EfficientFCParameters": EfficientFCParameters,
        "extract_features": extract_features,
        "impute": impute,
    }


class NumericFeatureSelectorMixin:
    def _fit_feature_space(self, X: pd.DataFrame, y: pd.Series, top_k: int = 140):
        numeric_cols = [
            c for c in X.columns
            if c not in ["ds", "y_new"] and pd.api.types.is_numeric_dtype(X[c])
        ]

        Xn = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        valid_cols = []
        for c in Xn.columns:
            notna_rate = Xn[c].notna().mean()
            uniq = Xn[c].nunique(dropna=True)
            if notna_rate < 0.80:
                continue
            if uniq <= 1:
                continue
            valid_cols.append(c)

        Xn = Xn[valid_cols]
        if Xn.shape[1] == 0:
            raise ValueError("利用可能な数値特徴量がありません。")

        medians = Xn.median(numeric_only=True)
        Xn = Xn.fillna(medians)

        corr = Xn.corrwith(pd.Series(y, index=Xn.index)).abs().fillna(0.0)
        selected = corr.sort_values(ascending=False).head(min(top_k, len(corr))).index.tolist()

        self.feature_cols_ = selected
        self.medians_ = medians[selected]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X[self.feature_cols_].replace([np.inf, -np.inf], np.nan)
        return Xn.fillna(self.medians_)


class GlobalRatioModel:
    name = "global_ratio"

    def __init__(self, config: BackcastConfig):
        self.config = config

    def fit(self, train_df: pd.DataFrame):
        tmp = train_df.copy()
        raw_ratio = tmp["y_new"] / np.clip(tmp["y_old"], 1.0, None)
        ql, qh = raw_ratio.quantile(self.config.ratio_clip_quantiles[0]), raw_ratio.quantile(self.config.ratio_clip_quantiles[1])
        self.global_ratio_ = float(raw_ratio.clip(ql, qh).median())
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        pred = pred_df["y_old"].to_numpy() * self.global_ratio_
        return np.clip(pred, 0, None)


class EmpiricalBayesDowRatioModel:
    name = "eb_dow_ratio"

    def __init__(self, config: BackcastConfig, group_cols: Optional[List[str]] = None, prior_strength: float = 8.0):
        self.config = config
        self.group_cols = group_cols or ["dow", "is_holiday", "is_month_turn"]
        self.prior_strength = prior_strength

    def fit(self, train_df: pd.DataFrame):
        tmp = train_df.copy()
        tmp["ratio"] = tmp["y_new"] / np.clip(tmp["y_old"], 1.0, None)
        ql, qh = tmp["ratio"].quantile(self.config.ratio_clip_quantiles[0]), tmp["ratio"].quantile(self.config.ratio_clip_quantiles[1])
        tmp["ratio"] = tmp["ratio"].clip(ql, qh)

        self.global_ratio_ = float(tmp["ratio"].median())
        grp = tmp.groupby(self.group_cols).agg(ratio_mean=("ratio", "mean"), n=("ratio", "size")).reset_index()
        grp["ratio_shrunk"] = (grp["n"] * grp["ratio_mean"] + self.prior_strength * self.global_ratio_) / (grp["n"] + self.prior_strength)
        self.map_ = {
            tuple(row[c] for c in self.group_cols): float(row["ratio_shrunk"])
            for _, row in grp.iterrows()
        }
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        ratios = []
        for _, row in pred_df.iterrows():
            key = tuple(row[c] for c in self.group_cols)
            ratio = self.map_.get(key, self.global_ratio_)
            ratios.append(ratio)
        pred = pred_df["y_old"].to_numpy() * np.asarray(ratios)
        return np.clip(pred, 0, None)


class LGBMDirectModel(NumericFeatureSelectorMixin):
    name = "lgbm_direct"

    def __init__(self, config: BackcastConfig, lib):
        self.config = config
        self.lib = lib

    def fit(self, train_df: pd.DataFrame):
        X = train_df.drop(columns=["y_new"])
        y = train_df["y_new"]
        self._fit_feature_space(X, y, top_k=self.config.top_k_features)
        Xtr = self._transform(X)

        self.model_ = self.lib["LGBMRegressor"](
            objective="regression",
            n_estimators=self.config.lgbm_n_estimators,
            learning_rate=self.config.lgbm_learning_rate,
            num_leaves=self.config.lgbm_num_leaves,
            max_depth=-1,
            min_child_samples=self.config.lgbm_min_child_samples,
            subsample=self.config.lgbm_subsample,
            colsample_bytree=self.config.lgbm_colsample_bytree,
            reg_alpha=self.config.lgbm_reg_alpha,
            reg_lambda=self.config.lgbm_reg_lambda,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.model_.fit(Xtr, y)
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(pred_df.drop(columns=["y_new"], errors="ignore"))
        pred = self.model_.predict(X)
        return np.clip(pred, 0, None)


class LGBMLogRatioModel(NumericFeatureSelectorMixin):
    name = "lgbm_logratio"

    def __init__(self, config: BackcastConfig, lib):
        self.config = config
        self.lib = lib

    def fit(self, train_df: pd.DataFrame):
        X = train_df.drop(columns=["y_new"])
        y = np.log1p(train_df["y_new"]) - np.log1p(train_df["y_old"])
        self._fit_feature_space(X, y, top_k=self.config.top_k_features)
        Xtr = self._transform(X)

        self.model_ = self.lib["LGBMRegressor"](
            objective="regression",
            n_estimators=self.config.lgbm_n_estimators,
            learning_rate=self.config.lgbm_learning_rate,
            num_leaves=self.config.lgbm_num_leaves,
            max_depth=-1,
            min_child_samples=self.config.lgbm_min_child_samples,
            subsample=self.config.lgbm_subsample,
            colsample_bytree=self.config.lgbm_colsample_bytree,
            reg_alpha=self.config.lgbm_reg_alpha,
            reg_lambda=self.config.lgbm_reg_lambda,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.model_.fit(Xtr, y)
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(pred_df.drop(columns=["y_new"], errors="ignore"))
        log_ratio_pred = self.model_.predict(X)
        pred = np.expm1(np.log1p(pred_df["y_old"].to_numpy()) + log_ratio_pred)
        return np.clip(pred, 0, None)


class ProphetRegressorModel:
    name = "prophet_regressor"

    def __init__(self, config: BackcastConfig, lib):
        self.config = config
        self.lib = lib
        self.reg_cols = [
            "y_old",
            "is_holiday",
            "is_day_before_holiday",
            "is_day_after_holiday",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_month_turn",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]

    def _make_holidays(self, start_ds, end_ds):
        rows = []
        jpholiday = self.lib["jpholiday"]
        for d in pd.date_range(start_ds, end_ds, freq="D"):
            holiday_name = jpholiday.is_holiday_name(d.date())
            if holiday_name:
                rows.append({"ds": d, "holiday": holiday_name})
        return pd.DataFrame(rows)

    def fit(self, train_df: pd.DataFrame):
        if not self.lib["prophet_ok"] or not self.config.use_prophet:
            self.enabled_ = False
            return self

        tr = train_df[["ds", "y_new"] + self.reg_cols].copy().rename(columns={"y_new": "y"})
        holidays_df = self._make_holidays(tr["ds"].min(), tr["ds"].max())
        Prophet = self.lib["Prophet"]
        try:
            self.model_ = Prophet(
                holidays=holidays_df,
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.8,
            )
            self.model_.add_seasonality(name="monthly", period=30.5, fourier_order=self.config.monthly_fourier_order)
            for c in self.reg_cols:
                self.model_.add_regressor(c, standardize=True)
            self.model_.fit(tr)
            self.enabled_ = True
        except Exception:
            self.enabled_ = False
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "enabled_", False):
            return np.full(len(pred_df), np.nan)
        future = pred_df[["ds"] + self.reg_cols].copy()
        fcst = self.model_.predict(future)
        return np.clip(fcst["yhat"].to_numpy(), 0, None)


class CalibratedBlendModel:
    def __init__(self, config: BackcastConfig, lib):
        self.config = config
        self.lib = lib
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: Optional[List[str]] = None
        self.isotonic_: Optional[object] = None

    def fit(self, oof_df: pd.DataFrame, model_names: List[str]):
        self.model_names_ = model_names
        valid = oof_df.dropna(subset=model_names + ["y_true"]).copy()
        X = valid[model_names].to_numpy(dtype=float)
        y = valid["y_true"].to_numpy(dtype=float)

        nnls = self.lib["nnls"]
        weights, _ = nnls(X, y)
        if weights.sum() == 0:
            weights = np.ones(len(model_names)) / len(model_names)
        else:
            weights = weights / weights.sum()
        self.weights_ = weights

        raw_pred = X @ self.weights_
        try:
            IsotonicRegression = self.lib["IsotonicRegression"]
            self.isotonic_ = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
            self.isotonic_.fit(raw_pred, y)
        except Exception:
            self.isotonic_ = None
        return self

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        X = pred_df[self.model_names_].to_numpy(dtype=float)
        raw = X @ self.weights_
        if self.isotonic_ is not None:
            raw = self.isotonic_.predict(raw)
        return np.clip(raw, 0, None)

    def get_weight_table(self) -> pd.DataFrame:
        return pd.DataFrame({"model": self.model_names_, "weight": self.weights_}).sort_values("weight", ascending=False)


class BackcastEnsemble:
    def __init__(self, config: Optional[BackcastConfig] = None):
        self.config = config or BackcastConfig()
        self.lib = _safe_import_dependencies()
        self.fitted_models_: Dict[str, object] = {}
        self.blender_: Optional[CalibratedBlendModel] = None
        self.feature_df_: Optional[pd.DataFrame] = None
        self.result_: Optional[pd.DataFrame] = None
        self.oof_: Optional[pd.DataFrame] = None
        self.cv_scores_: Optional[pd.DataFrame] = None
        self.ensemble_score_: Optional[pd.DataFrame] = None

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        need = {self.config.date_col, self.config.old_col, self.config.new_col}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"CSVに必要な列が不足しています: {miss}")

        df = df.rename(columns={
            self.config.date_col: "ds",
            self.config.old_col: "y_old",
            self.config.new_col: "y_new",
        }).copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["y_old"] = pd.to_numeric(df["y_old"], errors="coerce")
        df["y_new"] = pd.to_numeric(df["y_new"], errors="coerce")

        df = df.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)
        full_ds = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        df = pd.DataFrame({"ds": full_ds}).merge(df, on="ds", how="left")

        if df["y_old"].isna().any():
            null_n = int(df["y_old"].isna().sum())
            raise ValueError(f"y_old は全期間で必要です。欠損件数: {null_n}")
        return df

    def add_calendar_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        jpholiday = self.lib["jpholiday"]
        out = frame.copy()
        ds = out["ds"]

        out["dow"] = ds.dt.weekday
        out["dom"] = ds.dt.day
        out["month"] = ds.dt.month
        out["quarter"] = ds.dt.quarter
        out["weekofmonth"] = ((ds.dt.day - 1) // 7) + 1
        out["doy"] = ds.dt.dayofyear
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out["is_month_start"] = ds.dt.is_month_start.astype(int)
        out["is_month_end"] = ds.dt.is_month_end.astype(int)
        out["is_month_turn"] = ds.dt.day.isin([28, 29, 30, 31, 1, 2, 3]).astype(int)
        out["is_pay_cycle_like"] = ds.dt.day.isin([10, 25, 26, 27]).astype(int)

        out["is_holiday"] = ds.map(lambda x: int(jpholiday.is_holiday(x.date())))
        out["is_day_before_holiday"] = ds.map(lambda x: int(jpholiday.is_holiday((x + pd.Timedelta(days=1)).date())))
        out["is_day_after_holiday"] = ds.map(lambda x: int(jpholiday.is_holiday((x - pd.Timedelta(days=1)).date())))

        for col, period in [("dow", 7), ("dom", 31), ("month", 12), ("doy", 366)]:
            out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
            out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)

        out["holiday_x_weekend"] = out["is_holiday"] * out["is_weekend"]
        out["month_end_x_dow"] = out["is_month_end"] * out["dow"]
        return out

    def add_y_old_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        col = "y_old"
        lags = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

        base = out[col].shift(1)
        for w in [3, 7, 14, 28, 56]:
            out[f"{col}_roll_mean_{w}"] = base.rolling(w, min_periods=1).mean()
            out[f"{col}_roll_std_{w}"] = base.rolling(w, min_periods=2).std()
            out[f"{col}_roll_min_{w}"] = base.rolling(w, min_periods=1).min()
            out[f"{col}_roll_max_{w}"] = base.rolling(w, min_periods=1).max()
            out[f"{col}_roll_median_{w}"] = base.rolling(w, min_periods=1).median()

        out[f"{col}_diff_1"] = out[col] - out[col].shift(1)
        out[f"{col}_diff_7"] = out[col] - out[col].shift(7)
        out[f"{col}_diff_28"] = out[col] - out[col].shift(28)
        out[f"{col}_ratio_7"] = out[col] / (out[col].shift(7) + EPS)
        out[f"{col}_ratio_28"] = out[col] / (out[col].shift(28) + EPS)
        out[f"{col}_ma_ratio_7_28"] = out[f"{col}_roll_mean_7"] / (out[f"{col}_roll_mean_28"] + EPS)
        out[f"{col}_z_28"] = (out[col] - out[f"{col}_roll_mean_28"]) / (out[f"{col}_roll_std_28"] + EPS)
        return out

    def add_overlap_mapping_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        obs = out[out["y_new"].notna()].copy()
        if len(obs) == 0:
            out["hist_ratio_dow"] = np.nan
            out["hist_ratio_holiday"] = np.nan
            out["hist_ratio_global"] = np.nan
            return out

        obs["ratio"] = obs["y_new"] / np.clip(obs["y_old"], 1.0, None)
        ql, qh = obs["ratio"].quantile(self.config.ratio_clip_quantiles[0]), obs["ratio"].quantile(self.config.ratio_clip_quantiles[1])
        obs["ratio"] = obs["ratio"].clip(ql, qh)

        global_ratio = float(obs["ratio"].median())
        dow_map = obs.groupby("dow")["ratio"].median().to_dict()
        holiday_map = obs.groupby("is_holiday")["ratio"].median().to_dict()

        out["hist_ratio_global"] = global_ratio
        out["hist_ratio_dow"] = out["dow"].map(dow_map).fillna(global_ratio)
        out["hist_ratio_holiday"] = out["is_holiday"].map(holiday_map).fillna(global_ratio)
        out["hist_ynew_proxy_global"] = out["y_old"] * out["hist_ratio_global"]
        out["hist_ynew_proxy_dow"] = out["y_old"] * out["hist_ratio_dow"]
        out["hist_ynew_proxy_holiday"] = out["y_old"] * out["hist_ratio_holiday"]
        return out

    def build_tsfresh_features(self, frame: pd.DataFrame, window: int) -> pd.DataFrame:
        extract_features = self.lib["extract_features"]
        EfficientFCParameters = self.lib["EfficientFCParameters"]
        impute = self.lib["impute"]

        vals = frame["y_old"].to_numpy(dtype=float)
        dss = frame["ds"].to_numpy()
        records = []
        for i in range(window - 1, len(frame)):
            pred_ds = pd.Timestamp(dss[i])
            hist = vals[i - window + 1:i + 1]
            for t, v in enumerate(hist):
                records.append((pred_ds, t, float(v)))
        long_df = pd.DataFrame(records, columns=["id", "time", "value"])

        feats = extract_features(
            long_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=EfficientFCParameters(),
            impute_function=impute,
            disable_progressbar=True,
            n_jobs=0,
        )
        feats = feats.add_prefix(f"tsf_{window}_").reset_index()
        if "index" in feats.columns:
            feats = feats.rename(columns={"index": "ds"})
        if f"tsf_{window}_id" in feats.columns:
            feats = feats.rename(columns={f"tsf_{window}_id": "ds"})
        feats["ds"] = pd.to_datetime(feats["ds"])
        return feats

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = self.add_calendar_features(df)
        feat_df = self.add_y_old_features(feat_df)
        feat_df = self.add_overlap_mapping_features(feat_df)

        for w in self.config.tsfresh_windows:
            tsf = self.build_tsfresh_features(df[["ds", "y_old"]].copy(), window=w)
            feat_df = feat_df.merge(tsf, on="ds", how="left")

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        return feat_df

    def _get_model_classes(self):
        models = [
            GlobalRatioModel(self.config),
            EmpiricalBayesDowRatioModel(self.config),
            LGBMDirectModel(self.config, self.lib),
            LGBMLogRatioModel(self.config, self.lib),
        ]
        if self.config.use_prophet:
            models.append(ProphetRegressorModel(self.config, self.lib))
        return models

    def fit(self, df: pd.DataFrame):
        feat_df = self.build_features(df)
        self.feature_df_ = feat_df.copy()

        train_df = feat_df[feat_df["y_new"].notna()].copy().reset_index(drop=True)
        if len(train_df) < max(self.config.test_size * 2, 40):
            raise ValueError("y_new の重複期間が短すぎます。少なくとも40行以上を推奨します。")

        TimeSeriesSplit = self.lib["TimeSeriesSplit"]
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=self.config.test_size,
            gap=self.config.gap,
        )

        models = self._get_model_classes()
        model_names = [m.name for m in models]
        oof = pd.DataFrame({"ds": train_df["ds"], "y_true": train_df["y_new"]})
        for name in model_names:
            oof[name] = np.nan

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_df), start=1):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            print(f"[CV fold {fold}] train={tr['ds'].min().date()}〜{tr['ds'].max().date()} valid={va['ds'].min().date()}〜{va['ds'].max().date()} n_train={len(tr)} n_valid={len(va)}")
            for model in models:
                fitted = model.fit(tr)
                pred = fitted.predict(va)
                oof.loc[va_idx, model.name] = pred

        valid_pred_cols = []
        for c in model_names:
            nan_ratio = oof[c].isna().mean()
            if nan_ratio < 0.5:
                valid_pred_cols.append(c)

        oof_valid = oof.dropna(subset=valid_pred_cols + ["y_true"]).copy()
        self.oof_ = oof_valid.copy()

        score_table = []
        for c in valid_pred_cols:
            score_table.append(metric_frame(oof_valid["y_true"], oof_valid[c], c))
        self.cv_scores_ = pd.concat(score_table, ignore_index=True).sort_values("WAPE")

        blender = CalibratedBlendModel(self.config, self.lib)
        blender.fit(oof_valid, valid_pred_cols)
        oof_valid["ensemble"] = blender.predict(oof_valid[valid_pred_cols])
        self.ensemble_score_ = metric_frame(oof_valid["y_true"], oof_valid["ensemble"], "ensemble")
        self.blender_ = blender

        self.fitted_models_ = {}
        for model in models:
            fitted = model.fit(train_df)
            pred_all = fitted.predict(feat_df)
            self.fitted_models_[model.name] = {"model": fitted, "pred_all": pred_all}

        pred_table = pd.DataFrame({"ds": feat_df["ds"]})
        for name, bundle in self.fitted_models_.items():
            pred_table[name] = bundle["pred_all"]

        use_names = self.blender_.model_names_
        pred_table["ensemble_raw"] = pred_table[use_names].to_numpy() @ self.blender_.weights_
        pred_table["ensemble"] = self.blender_.predict(pred_table[use_names])

        result = feat_df[["ds", "y_old", "y_new"]].merge(pred_table, on="ds", how="left")
        result["y_new_filled"] = result["y_new"]
        mask_missing = result["y_new_filled"].isna()
        result.loc[mask_missing, "y_new_filled"] = result.loc[mask_missing, "ensemble"]

        resid = oof_valid["y_true"].to_numpy() - oof_valid["ensemble"].to_numpy()
        sigma = float(np.nanstd(resid))
        result["y_new_lower80"] = np.where(mask_missing, np.clip(result["ensemble"] - 1.2816 * sigma, 0, None), result["y_new"])
        result["y_new_upper80"] = np.where(mask_missing, np.clip(result["ensemble"] + 1.2816 * sigma, 0, None), result["y_new"])

        self.result_ = result
        return self

    def get_weight_table(self) -> pd.DataFrame:
        if self.blender_ is None:
            raise RuntimeError("fit 後に実行してください。")
        return self.blender_.get_weight_table()

    def save_outputs(self, outdir: str | Path):
        if self.result_ is None:
            raise RuntimeError("fit 後に save_outputs を呼んでください。")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        backcast_only = self.result_[self.result_["y_new"].isna()][
            ["ds", "y_old", "ensemble", "y_new_lower80", "y_new_upper80"]
        ].rename(columns={"ensemble": "y_new_imputed"})

        self.result_.to_csv(outdir / "y_new_filled_full.csv", index=False, encoding="utf-8-sig")
        backcast_only.to_csv(outdir / "y_new_backcast_only.csv", index=False, encoding="utf-8-sig")
        self.cv_scores_.to_csv(outdir / "cv_model_scores.csv", index=False, encoding="utf-8-sig")
        self.ensemble_score_.to_csv(outdir / "cv_ensemble_score.csv", index=False, encoding="utf-8-sig")
        self.get_weight_table().to_csv(outdir / "ensemble_weights.csv", index=False, encoding="utf-8-sig")

        summary = {
            "config": asdict(self.config),
            "cv_scores": self.cv_scores_.to_dict(orient="records"),
            "ensemble_score": self.ensemble_score_.to_dict(orient="records"),
            "weights": self.get_weight_table().to_dict(orient="records"),
        }
        with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return outdir


def make_requirements_txt() -> str:
    return "\n".join([
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "prophet>=1.1.5",
        "tsfresh>=0.20.0",
        "jpholiday>=1.0.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ]) + "\n"


def make_notebook_example() -> str:
    return '''# JupyterLab example\nfrom callcenter_ynew_backcast_ensemble import BackcastConfig, BackcastEnsemble\n\nconfig = BackcastConfig(\n    n_splits=3,\n    test_size=21,\n    gap=7,\n    tsfresh_windows=(28, 56),\n    top_k_features=140,\n    use_prophet=True,\n)\n\nrunner = BackcastEnsemble(config)\ndf = runner.load_csv("your_data.csv")\nrunner.fit(df)\nrunner.save_outputs("./backcast_output")\n\nprint(runner.cv_scores_)\nprint(runner.ensemble_score_)\nprint(runner.get_weight_table())\nrunner.result_.head()\n''' 


def main():
    parser = argparse.ArgumentParser(description="Backcast y_new from y_old using ensemble models.")
    parser.add_argument("--csv", required=True, help="Input CSV path with ds, y_old, y_new columns")
    parser.add_argument("--outdir", default="./backcast_output", help="Output directory")
    parser.add_argument("--date-col", default="ds")
    parser.add_argument("--old-col", default="y_old")
    parser.add_argument("--new-col", default="y_new")
    parser.add_argument("--disable-prophet", action="store_true")
    args = parser.parse_args()

    config = BackcastConfig(
        date_col=args.date_col,
        old_col=args.old_col,
        new_col=args.new_col,
        use_prophet=not args.disable_prophet,
    )

    runner = BackcastEnsemble(config)
    df = runner.load_csv(args.csv)
    runner.fit(df)
    outdir = runner.save_outputs(args.outdir)

    req_path = Path(outdir) / "requirements.txt"
    req_path.write_text(make_requirements_txt(), encoding="utf-8")
    nb_path = Path(outdir) / "notebook_example.py"
    nb_path.write_text(make_notebook_example(), encoding="utf-8")

    print("=== CV model scores ===")
    print(runner.cv_scores_.to_string(index=False))
    print("\n=== Ensemble score ===")
    print(runner.ensemble_score_.to_string(index=False))
    print("\n=== Ensemble weights ===")
    print(runner.get_weight_table().to_string(index=False))
    print(f"\nSaved to: {Path(outdir).resolve()}")


if __name__ == "__main__":
    main()
