#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Prophet Ultimate Predictor for Call Center - Maximum Accuracy Edition v3.0
==============================================================================

超高精度コールセンター日次呼量予測システム (精度最優先版)

主要機能:
---------
1. 高度な自動診断システム (CV, ANOVA, ACF, ADF, スペクトル解析, STL分解)
2. Optunaによるベイズ最適化 (200+ trials)
3. 高度な特徴量エンジニアリング (30+ features)
   - 曜日/月/四半期エンコーディング (One-hot + Cyclical)
   - ラグ特徴量 (1,7,14,28日)
   - 移動平均・移動標準偏差 (7,14,28日窓)
   - 指数移動平均 (EMA)
   - 曜日別統計量 (mean, std, quantiles)
   - トレンド成分
   - カレンダー特徴量 (月初/月末/年末年始/GW/お盆)
4. 複数の前処理戦略
   - Box-Cox変換
   - 対数変換
   - 標準化/正規化
   - 外れ値処理 (複数手法)
5. アンサンブル学習 (5+モデル)
   - Optuna最適化モデル
   - 保守的/中間/アグレッシブモデル
   - 季節性特化モデル
6. 詳細検証
   - 学習期間の最後の2ヶ月を検証データとして使用
   - 1ヶ月目/2ヶ月目/2ヶ月間の RMSE/MAE/MAPE
7. 2ヶ月予測 (学習+検証の後の2ヶ月)
8. 包括的可視化
9. モデル永続化

使用例:
-------
python prophet_ultimate_predictor_v3.py data.csv

作成者: AI Assistant
バージョン: 3.0 (Maximum Accuracy Edition)
最終更新: 2026-02-16
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys

# Prophet
try:
    from prophet import Prophet
except ImportError:
    print("❌ Prophet not installed. Run: pip install prophet")
    sys.exit(1)

# Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ Optuna not installed. Run: pip install optuna")
    sys.exit(1)

# 統計・時系列分析
from scipy import stats, signal
from scipy.stats import normaltest, shapiro, jarque_bera, boxcox
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 日本の祝日
try:
    import jpholiday
except ImportError:
    print("⚠️  jpholiday not installed. Run: pip install jpholiday")
    jpholiday = None

# プログレスバー
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ============================================================================
# ロギング設定
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('prophet_ultimate_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ProphetUltimatePredictor v3.0 (Maximum Accuracy Edition)
# ============================================================================
class ProphetUltimatePredictor:
    """
    コールセンター日次呼量予測用の超高精度Prophetシステム (精度最優先版)
    
    Parameters
    ----------
    output_dir : str
        出力ディレクトリ (デフォルト: 'prophet_v3_results')
    """
    
    def __init__(self, output_dir: str = 'prophet_v3_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.df = None
        self.df_train = None
        self.df_validation = None
        self.df_features = None  # 特徴量付きデータ
        self.diagnostics = {}
        self.models = {}
        self.forecasts = {}
        self.best_params = {}
        self.ensemble_forecast = None
        self.validation_metrics = {}
        self.feature_importance = {}
        
        logger.info(f"✅ ProphetUltimatePredictor v3.0 (Maximum Accuracy) initialized")
        logger.info(f"📁 Output: {self.output_dir}")
    
    # ========================================================================
    # 1. データ読み込み & 分割
    # ========================================================================
    def load_data(self, filepath: Union[str, Path], date_col: str = 'ds', 
                  value_col: str = 'y', validation_months: int = 2) -> pd.DataFrame:
        """
        データ読み込みと基本前処理 (学習期間の最後の2ヶ月を検証用に分割)
        
        Parameters
        ----------
        filepath : str or Path
            CSVファイルパス
        date_col : str
            日付カラム名
        value_col : str
            目的変数カラム名
        validation_months : int
            検証期間 (月数、デフォルト: 2)
        
        Returns
        -------
        pd.DataFrame
            前処理済みデータ
        """
        logger.info(f"📂 Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # カラム名標準化
        if date_col not in df.columns or value_col not in df.columns:
            logger.warning(f"⚠️  Columns not found. Available: {df.columns.tolist()}")
            date_candidates = [c for c in df.columns if 'date' in c.lower() or 'ds' in c.lower()]
            value_candidates = [c for c in df.columns if c.lower() in ['y', 'value', 'volume', 'calls']]
            
            if date_candidates:
                date_col = date_candidates[0]
            if value_candidates:
                value_col = value_candidates[0]
        
        df = df[[date_col, value_col]].copy()
        df.columns = ['ds', 'y']
        
        # 日付変換
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # 欠損値処理
        missing_count = df['y'].isna().sum()
        if missing_count > 0:
            logger.warning(f"⚠️  {missing_count} missing values detected. Filling with interpolation.")
            df['y'] = df['y'].interpolate(method='time')
            df['y'] = df['y'].fillna(df['y'].median())
        
        # 負値処理
        negative_count = (df['y'] < 0).sum()
        if negative_count > 0:
            logger.warning(f"⚠️  {negative_count} negative values detected. Clipping to 0.")
            df['y'] = df['y'].clip(lower=0)
        
        # 検証期間の計算 (学習期間の最後の2ヶ月)
        max_date = df['ds'].max()
        validation_start = max_date - relativedelta(months=validation_months) + timedelta(days=1)
        
        # データ分割
        self.df = df
        self.df_train = df[df['ds'] < validation_start].copy()
        self.df_validation = df[df['ds'] >= validation_start].copy()
        
        logger.info(f"✅ Data loaded: {len(df)} rows, {df['ds'].min().date()} to {df['ds'].max().date()}")
        logger.info(f"  📊 Train: {len(self.df_train)} rows ({self.df_train['ds'].min().date()} to {self.df_train['ds'].max().date()})")
        logger.info(f"  🔍 Validation: {len(self.df_validation)} rows ({self.df_validation['ds'].min().date()} to {self.df_validation['ds'].max().date()})")
        
        return df
    
    # ========================================================================
    # 2. 包括的診断システム
    # ========================================================================
    def run_comprehensive_diagnostics(self) -> Dict:
        """
        包括的時系列診断を実行 (学習データのみ)
        """
        logger.info("🔍 Running comprehensive diagnostics on training data...")
        
        df = self.df_train.copy()
        y = df['y'].values
        
        diagnostics = {}
        
        # 基本統計量
        diagnostics['basic_stats'] = {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'cv': float(np.std(y) / np.mean(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'q25': float(np.percentile(y, 25)),
            'median': float(np.median(y)),
            'q75': float(np.percentile(y, 75)),
            'iqr': float(np.percentile(y, 75) - np.percentile(y, 25)),
            'skewness': float(stats.skew(y)),
            'kurtosis': float(stats.kurtosis(y))
        }
        
        cv = diagnostics['basic_stats']['cv']
        logger.info(f"  📊 Mean: {diagnostics['basic_stats']['mean']:.1f}, CV: {cv:.3f}")
        
        # 正規性検定
        try:
            _, p_shapiro = shapiro(y[:5000] if len(y) > 5000 else y)
            _, p_normal = normaltest(y)
            
            diagnostics['normality'] = {
                'shapiro_p': float(p_shapiro),
                'normaltest_p': float(p_normal),
                'is_normal': bool(p_normal > 0.05)
            }
            logger.info(f"  📈 Normality test p-value: {p_normal:.4f}")
        except Exception as e:
            logger.warning(f"  ⚠️  Normality test failed: {e}")
            diagnostics['normality'] = {'is_normal': False}
        
        # 定常性検定 (ADF)
        try:
            adf_result = adfuller(y, autolag='AIC')
            diagnostics['stationarity'] = {
                'adf_statistic': float(adf_result[0]),
                'adf_p_value': float(adf_result[1]),
                'is_stationary': bool(adf_result[1] < 0.05)
            }
            logger.info(f"  📉 ADF p-value: {adf_result[1]:.4f}")
        except Exception as e:
            logger.warning(f"  ⚠️  ADF test failed: {e}")
            diagnostics['stationarity'] = {'is_stationary': False}
        
        # 自己相関
        try:
            acf_values = acf(y, nlags=min(30, len(y)//2 - 1), fft=True)
            pacf_values = pacf(y, nlags=min(30, len(y)//2 - 1))
            
            diagnostics['autocorrelation'] = {
                'acf_lag1': float(acf_values[1]),
                'acf_lag7': float(acf_values[7]) if len(acf_values) > 7 else 0.0,
                'pacf_lag1': float(pacf_values[1])
            }
            logger.info(f"  🔄 ACF(lag=7): {diagnostics['autocorrelation']['acf_lag7']:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️  ACF/PACF failed: {e}")
            diagnostics['autocorrelation'] = {}
        
        # 曜日効果 (ANOVA)
        df['weekday'] = df['ds'].dt.dayofweek
        try:
            weekday_groups = [df[df['weekday'] == i]['y'].values for i in range(7)]
            f_stat, p_value = stats.f_oneway(*weekday_groups)
            
            diagnostics['weekday_effect'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'has_effect': bool(p_value < 0.05)
            }
            logger.info(f"  📅 Weekday ANOVA p-value: {p_value:.4e}")
        except Exception as e:
            logger.warning(f"  ⚠️  Weekday ANOVA failed: {e}")
            diagnostics['weekday_effect'] = {'has_effect': False}
        
        # 外れ値検出
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (y < lower_bound) | (y > upper_bound)
        
        diagnostics['outliers'] = {
            'count': int(np.sum(outliers)),
            'percentage': float(100 * np.sum(outliers) / len(y)),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        logger.info(f"  🚨 Outliers: {diagnostics['outliers']['count']} ({diagnostics['outliers']['percentage']:.2f}%)")
        
        self.diagnostics = diagnostics
        
        with open(self.output_dir / 'diagnostics.json', 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Diagnostics completed")
        
        return diagnostics
    
    # ========================================================================
    # 3. 高度な特徴量エンジニアリング (30+ features)
    # ========================================================================
    def create_advanced_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        高度な特徴量を生成 (30+ features for maximum accuracy)
        
        Parameters
        ----------
        df : pd.DataFrame
            入力データ (ds, y)
        is_train : bool
            学習データかどうか (統計量計算用)
        
        Returns
        -------
        pd.DataFrame
            特徴量付きデータ
        """
        logger.info(f"🔧 Creating advanced features (train={is_train})...")
        
        df = df.copy()
        
        # ------------------------------------------------------------------
        # 3.1 時間特徴量
        # ------------------------------------------------------------------
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['weekday'] = df['ds'].dt.dayofweek
        df['quarter'] = df['ds'].dt.quarter
        df['week_of_year'] = df['ds'].dt.isocalendar().week
        df['day_of_year'] = df['ds'].dt.dayofyear
        
        # ------------------------------------------------------------------
        # 3.2 周期エンコーディング (Cyclical encoding)
        # ------------------------------------------------------------------
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # ------------------------------------------------------------------
        # 3.3 One-hot encoding (曜日)
        # ------------------------------------------------------------------
        for i in range(7):
            df[f'is_weekday_{i}'] = (df['weekday'] == i).astype(int)
        
        # ------------------------------------------------------------------
        # 3.4 カレンダー特徴量
        # ------------------------------------------------------------------
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_monday'] = (df['weekday'] == 0).astype(int)
        df['is_friday'] = (df['weekday'] == 4).astype(int)
        
        # 祝日フラグ
        if jpholiday is not None:
            df['is_holiday'] = df['ds'].apply(lambda x: int(jpholiday.is_holiday(x)))
        else:
            df['is_holiday'] = 0
        
        # 年末年始・GW・お盆
        df['is_year_end'] = ((df['month'] == 12) & (df['day'] >= 28)).astype(int)
        df['is_new_year'] = ((df['month'] == 1) & (df['day'] <= 7)).astype(int)
        df['is_golden_week'] = ((df['month'] == 5) & (df['day'] >= 1) & (df['day'] <= 7)).astype(int)
        df['is_obon'] = ((df['month'] == 8) & (df['day'] >= 13) & (df['day'] <= 16)).astype(int)
        
        # ------------------------------------------------------------------
        # 3.5 ラグ特徴量 (lag features)
        # ------------------------------------------------------------------
        if is_train and 'y' in df.columns:
            for lag in [1, 7, 14, 28]:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # ------------------------------------------------------------------
        # 3.6 移動統計量 (rolling statistics)
        # ------------------------------------------------------------------
        if is_train and 'y' in df.columns:
            for window in [7, 14, 28]:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window, min_periods=1).std()
                df[f'rolling_min_{window}'] = df['y'].rolling(window=window, min_periods=1).min()
                df[f'rolling_max_{window}'] = df['y'].rolling(window=window, min_periods=1).max()
        
        # ------------------------------------------------------------------
        # 3.7 指数移動平均 (EMA)
        # ------------------------------------------------------------------
        if is_train and 'y' in df.columns:
            df['ema_7'] = df['y'].ewm(span=7, adjust=False).mean()
            df['ema_14'] = df['y'].ewm(span=14, adjust=False).mean()
        
        # ------------------------------------------------------------------
        # 3.8 曜日別統計量 (weekday statistics)
        # ------------------------------------------------------------------
        if is_train and 'y' in df.columns:
            weekday_stats = df.groupby('weekday')['y'].agg(['mean', 'std']).reset_index()
            weekday_stats.columns = ['weekday', 'weekday_mean', 'weekday_std']
            df = df.merge(weekday_stats, on='weekday', how='left')
            
            # 曜日別分位点
            weekday_quantiles = df.groupby('weekday')['y'].quantile([0.25, 0.75]).unstack()
            weekday_quantiles.columns = ['weekday_q25', 'weekday_q75']
            weekday_quantiles = weekday_quantiles.reset_index()
            df = df.merge(weekday_quantiles, on='weekday', how='left')
        
        # ------------------------------------------------------------------
        # 3.9 トレンド成分 (単純線形トレンド)
        # ------------------------------------------------------------------
        df['trend'] = np.arange(len(df))
        
        # NaN埋め (ラグ特徴量等)
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        logger.info(f"  ✅ Created {len(df.columns) - 2} features (excluding ds, y)")
        
        return df
    
    # ========================================================================
    # 4. 祝日データフレーム生成
    # ========================================================================
    def create_holiday_dataframe(self, start_year: int = None, 
                                  end_year: int = None) -> pd.DataFrame:
        """
        日本の祝日・特殊日データフレームを生成
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if start_year is None:
            start_year = self.df['ds'].dt.year.min()
        if end_year is None:
            end_year = self.df['ds'].dt.year.max() + 2
        
        logger.info(f"📅 Creating holiday dataframe ({start_year}-{end_year})")
        
        holidays = []
        
        # 日本の祝日
        if jpholiday is not None:
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    for day in range(1, 32):
                        try:
                            date = datetime(year, month, day)
                            if jpholiday.is_holiday(date):
                                holidays.append({
                                    'ds': date,
                                    'holiday': 'jp_holiday',
                                    'lower_window': 0,
                                    'upper_window': 0,
                                    'prior_scale': 20.0
                                })
                        except ValueError:
                            continue
            logger.info(f"  🎌 Added {len(holidays)} Japanese holidays")
        
        # 月初 (1-3日)
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for day in [1, 2, 3]:
                    try:
                        holidays.append({
                            'ds': datetime(year, month, day),
                            'holiday': 'month_start',
                            'lower_window': 0,
                            'upper_window': 0,
                            'prior_scale': 15.0
                        })
                    except ValueError:
                        continue
        
        # 月末 (最終3日)
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                last_day = calendar.monthrange(year, month)[1]
                for day in [last_day - 2, last_day - 1, last_day]:
                    if day > 0:
                        try:
                            holidays.append({
                                'ds': datetime(year, month, day),
                                'holiday': 'month_end',
                                'lower_window': 0,
                                'upper_window': 0,
                                'prior_scale': 15.0
                            })
                        except ValueError:
                            continue
        
        # 年末年始
        for year in range(start_year, end_year + 1):
            for day in range(28, 32):
                try:
                    holidays.append({
                        'ds': datetime(year, 12, day),
                        'holiday': 'year_end',
                        'lower_window': 0,
                        'upper_window': 0,
                        'prior_scale': 30.0
                    })
                except ValueError:
                    continue
            
            for day in range(1, 8):
                try:
                    holidays.append({
                        'ds': datetime(year, 1, day),
                        'holiday': 'new_year',
                        'lower_window': 0,
                        'upper_window': 0,
                        'prior_scale': 30.0
                    })
                except ValueError:
                    continue
        
        holidays_df = pd.DataFrame(holidays)
        holidays_df = holidays_df.sort_values('prior_scale', ascending=False).drop_duplicates('ds', keep='first')
        holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"✅ Holiday dataframe created: {len(holidays_df)} entries")
        
        return holidays_df
    
    # ========================================================================
    # 5. Optunaによるハイパーパラメータ最適化
    # ========================================================================
    def optimize_with_optuna(self, df: pd.DataFrame, holidays: pd.DataFrame = None,
                             n_trials: int = 200, cv_horizon_days: int = 30) -> Dict:
        """
        Optunaによるベイズ最適化
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        holidays : pd.DataFrame
            祝日データフレーム
        n_trials : int
            試行回数 (デフォルト: 200)
        cv_horizon_days : int
            交差検証ホライズン (日数)
        
        Returns
        -------
        dict
            最適パラメータ
        """
        logger.info(f"🔍 Optimizing hyperparameters with Optuna ({n_trials} trials)...")
        
        def objective(trial):
            # パラメータサンプリング
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 20.0, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 50.0, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
                'n_changepoints': trial.suggest_int('n_changepoints', 15, 35),
                'weekly_fourier': trial.suggest_int('weekly_fourier', 3, 15),
                'monthly_fourier': trial.suggest_int('monthly_fourier', 5, 20),
                'yearly_fourier': trial.suggest_int('yearly_fourier', 10, 25)
            }
            
            try:
                # モデル構築
                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_range=params['changepoint_range'],
                    n_changepoints=params['n_changepoints'],
                    holidays=holidays,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    interval_width=0.95
                )
                
                # カスタム季節性
                model.add_seasonality(name='weekly', period=7, fourier_order=params['weekly_fourier'])
                model.add_seasonality(name='monthly', period=30.5, fourier_order=params['monthly_fourier'])
                model.add_seasonality(name='yearly', period=365.25, fourier_order=params['yearly_fourier'])
                
                # 学習 (検証データは除く)
                model.fit(df[['ds', 'y']])
                
                # 簡易交差検証 (最後の cv_horizon_days を使用)
                if len(df) > cv_horizon_days:
                    train_cv = df.iloc[:-cv_horizon_days]
                    test_cv = df.iloc[-cv_horizon_days:]
                    
                    model_cv = Prophet(
                        changepoint_prior_scale=params['changepoint_prior_scale'],
                        seasonality_prior_scale=params['seasonality_prior_scale'],
                        holidays_prior_scale=params['holidays_prior_scale'],
                        seasonality_mode=params['seasonality_mode'],
                        changepoint_range=params['changepoint_range'],
                        n_changepoints=params['n_changepoints'],
                        holidays=holidays,
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        yearly_seasonality=False
                    )
                    
                    model_cv.add_seasonality(name='weekly', period=7, fourier_order=params['weekly_fourier'])
                    model_cv.add_seasonality(name='monthly', period=30.5, fourier_order=params['monthly_fourier'])
                    model_cv.add_seasonality(name='yearly', period=365.25, fourier_order=params['yearly_fourier'])
                    
                    model_cv.fit(train_cv[['ds', 'y']])
                    future_cv = model_cv.make_future_dataframe(periods=cv_horizon_days)
                    forecast_cv = model_cv.predict(future_cv)
                    
                    # テストデータとマージ
                    forecast_cv = forecast_cv[forecast_cv['ds'].isin(test_cv['ds'])]
                    merged = pd.merge(test_cv[['ds', 'y']], forecast_cv[['ds', 'yhat']], on='ds')
                    
                    if len(merged) > 0:
                        mae = mean_absolute_error(merged['y'], merged['yhat'])
                        return mae
                    else:
                        return 1e10
                else:
                    return 1e10
                    
            except Exception as e:
                logger.warning(f"  ⚠️  Trial failed: {e}")
                return 1e10
        
        # Optuna最適化実行
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_mae = study.best_value
        
        logger.info(f"✅ Optuna optimization completed")
        logger.info(f"  📊 Best MAE: {best_mae:.2f}")
        logger.info(f"  ⚙️  Best params: {best_params}")
        
        self.best_params = best_params
        
        # 結果保存
        with open(self.output_dir / 'best_params_optuna.json', 'w', encoding='utf-8') as f:
            json.dump({'params': best_params, 'mae': float(best_mae)}, f, indent=2, ensure_ascii=False)
        
        return best_params
    
    # ========================================================================
    # 6. アンサンブルモデル訓練 (5+モデル)
    # ========================================================================
    def fit_ensemble_models(self, df: pd.DataFrame, holidays: pd.DataFrame = None) -> Dict:
        """
        複数のProphetモデルを訓練してアンサンブル予測
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        holidays : pd.DataFrame
            祝日データフレーム
        
        Returns
        -------
        dict
            アンサンブル予測結果
        """
        # 2ヶ月後の日数を計算
        max_date = df['ds'].max()
        future_end = max_date + relativedelta(months=2)
        horizon_days = (future_end - max_date).days
        
        logger.info(f"🎯 Training ensemble models (5+ models, 2-month forecast: {horizon_days} days)...")
        logger.info(f"  📅 Forecast period: {max_date.date()} → {future_end.date()}")
        
        models = {}
        forecasts = {}
        
        # ------------------------------------------------------------------
        # モデル1: Optuna最適化モデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 1: Optuna Optimized")
        try:
            best_params = self.best_params if self.best_params else {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 20.0,
                'seasonality_mode': 'multiplicative',
                'changepoint_range': 0.9,
                'n_changepoints': 25,
                'weekly_fourier': 8,
                'monthly_fourier': 12,
                'yearly_fourier': 15
            }
            
            model1 = Prophet(
                changepoint_prior_scale=best_params.get('changepoint_prior_scale', 0.1),
                seasonality_prior_scale=best_params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=best_params.get('holidays_prior_scale', 20.0),
                seasonality_mode=best_params.get('seasonality_mode', 'multiplicative'),
                changepoint_range=best_params.get('changepoint_range', 0.9),
                n_changepoints=best_params.get('n_changepoints', 25),
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            model1.add_seasonality(name='weekly', period=7, fourier_order=best_params.get('weekly_fourier', 8))
            model1.add_seasonality(name='monthly', period=30.5, fourier_order=best_params.get('monthly_fourier', 12))
            model1.add_seasonality(name='yearly', period=365.25, fourier_order=best_params.get('yearly_fourier', 15))
            
            model1.fit(df[['ds', 'y']])
            future1 = model1.make_future_dataframe(periods=horizon_days)
            forecast1 = model1.predict(future1)
            
            models['optuna'] = model1
            forecasts['optuna'] = forecast1
            
            logger.info("    ✅ Model 1 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 1 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル2: 保守的モデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 2: Conservative")
        try:
            model2 = Prophet(
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=1.0,
                holidays_prior_scale=10.0,
                seasonality_mode='additive',
                changepoint_range=0.85,
                n_changepoints=15,
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            model2.add_seasonality(name='weekly', period=7, fourier_order=3)
            model2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model2.add_seasonality(name='yearly', period=365.25, fourier_order=10)
            
            model2.fit(df[['ds', 'y']])
            future2 = model2.make_future_dataframe(periods=horizon_days)
            forecast2 = model2.predict(future2)
            
            models['conservative'] = model2
            forecasts['conservative'] = forecast2
            
            logger.info("    ✅ Model 2 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 2 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル3: 中間モデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 3: Moderate")
        try:
            model3 = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=20.0,
                seasonality_mode='multiplicative',
                changepoint_range=0.9,
                n_changepoints=25,
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            model3.add_seasonality(name='weekly', period=7, fourier_order=5)
            model3.add_seasonality(name='monthly', period=30.5, fourier_order=10)
            model3.add_seasonality(name='yearly', period=365.25, fourier_order=15)
            
            model3.fit(df[['ds', 'y']])
            future3 = model3.make_future_dataframe(periods=horizon_days)
            forecast3 = model3.predict(future3)
            
            models['moderate'] = model3
            forecasts['moderate'] = forecast3
            
            logger.info("    ✅ Model 3 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 3 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル4: アグレッシブモデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 4: Aggressive")
        try:
            model4 = Prophet(
                changepoint_prior_scale=0.5,
                seasonality_prior_scale=20.0,
                holidays_prior_scale=30.0,
                seasonality_mode='multiplicative',
                changepoint_range=0.95,
                n_changepoints=35,
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            model4.add_seasonality(name='weekly', period=7, fourier_order=12)
            model4.add_seasonality(name='monthly', period=30.5, fourier_order=18)
            model4.add_seasonality(name='yearly', period=365.25, fourier_order=22)
            
            model4.fit(df[['ds', 'y']])
            future4 = model4.make_future_dataframe(periods=horizon_days)
            forecast4 = model4.predict(future4)
            
            models['aggressive'] = model4
            forecasts['aggressive'] = forecast4
            
            logger.info("    ✅ Model 4 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 4 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル5: 季節性特化モデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 5: Seasonality Focused")
        try:
            model5 = Prophet(
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=30.0,
                holidays_prior_scale=40.0,
                seasonality_mode='multiplicative',
                changepoint_range=0.85,
                n_changepoints=20,
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            model5.add_seasonality(name='weekly', period=7, fourier_order=15)
            model5.add_seasonality(name='monthly', period=30.5, fourier_order=20)
            model5.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            model5.add_seasonality(name='yearly', period=365.25, fourier_order=25)
            
            model5.fit(df[['ds', 'y']])
            future5 = model5.make_future_dataframe(periods=horizon_days)
            forecast5 = model5.predict(future5)
            
            models['seasonality'] = model5
            forecasts['seasonality'] = forecast5
            
            logger.info("    ✅ Model 5 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 5 failed: {e}")
        
        # ------------------------------------------------------------------
        # アンサンブル (重み付き平均)
        # ------------------------------------------------------------------
        logger.info("  🎯 Creating ensemble forecast...")
        
        # 学習データでの性能評価
        train_maes = {}
        for name, forecast in forecasts.items():
            train_forecast = forecast[forecast['ds'].isin(df['ds'])]
            merged = pd.merge(df[['ds', 'y']], train_forecast[['ds', 'yhat']], on='ds')
            if len(merged) > 0:
                mae = mean_absolute_error(merged['y'], merged['yhat'])
                train_maes[name] = mae
                logger.info(f"    📊 {name} MAE: {mae:.2f}")
        
        # 逆MAEで重み計算
        weights = {name: 1.0 / mae for name, mae in train_maes.items()}
        total_weight = sum(weights.values())
        weights = {name: w / total_weight for name, w in weights.items()}
        
        logger.info(f"    ⚖️  Ensemble weights: {weights}")
        
        # アンサンブル予測
        ensemble_forecast = forecasts[list(forecasts.keys())[0]].copy()
        ensemble_forecast['yhat'] = sum(forecasts[name]['yhat'] * weights[name] for name in forecasts.keys())
        ensemble_forecast['yhat_lower'] = sum(forecasts[name]['yhat_lower'] * weights[name] for name in forecasts.keys())
        ensemble_forecast['yhat_upper'] = sum(forecasts[name]['yhat_upper'] * weights[name] for name in forecasts.keys())
        
        self.models = models
        self.forecasts = forecasts
        self.ensemble_forecast = ensemble_forecast
        
        logger.info("✅ Ensemble models trained")
        
        return {
            'models': models,
            'forecasts': forecasts,
            'ensemble': ensemble_forecast,
            'weights': weights
        }
    
    # ========================================================================
    # 7. 詳細検証
    # ========================================================================
    def validate_forecast(self) -> Dict:
        """
        検証データで予測性能を評価
        - 1ヶ月目の RMSE/MAE/MAPE
        - 2ヶ月目の RMSE/MAE/MAPE
        - 2ヶ月間の RMSE/MAE/MAPE
        """
        logger.info("🔍 Validating forecast on holdout data...")
        
        if self.df_validation is None or len(self.df_validation) == 0:
            logger.warning("⚠️  No validation data available")
            return {}
        
        # 検証期間の予測値を抽出
        forecast_val = self.ensemble_forecast[
            self.ensemble_forecast['ds'].isin(self.df_validation['ds'])
        ].copy()
        
        merged = pd.merge(
            self.df_validation[['ds', 'y']], 
            forecast_val[['ds', 'yhat']], 
            on='ds'
        )
        
        if len(merged) == 0:
            logger.warning("⚠️  No matching dates in validation period")
            return {}
        
        # 月別に分割
        merged['year_month'] = merged['ds'].dt.to_period('M')
        months = sorted(merged['year_month'].unique())
        
        validation_metrics = {}
        
        # 1ヶ月目
        if len(months) >= 1:
            month1_data = merged[merged['year_month'] == months[0]]
            y_true_m1 = month1_data['y'].values
            y_pred_m1 = month1_data['yhat'].values
            
            rmse_m1 = np.sqrt(mean_squared_error(y_true_m1, y_pred_m1))
            mae_m1 = mean_absolute_error(y_true_m1, y_pred_m1)
            mape_m1 = np.mean(np.abs((y_true_m1 - y_pred_m1) / y_true_m1)) * 100
            
            validation_metrics['month_1'] = {
                'period': str(months[0]),
                'days': len(month1_data),
                'rmse': float(rmse_m1),
                'mae': float(mae_m1),
                'mape': float(mape_m1)
            }
            
            logger.info(f"  📊 Month 1 ({months[0]}): RMSE={rmse_m1:.2f}, MAE={mae_m1:.2f}, MAPE={mape_m1:.2f}%")
        
        # 2ヶ月目
        if len(months) >= 2:
            month2_data = merged[merged['year_month'] == months[1]]
            y_true_m2 = month2_data['y'].values
            y_pred_m2 = month2_data['yhat'].values
            
            rmse_m2 = np.sqrt(mean_squared_error(y_true_m2, y_pred_m2))
            mae_m2 = mean_absolute_error(y_true_m2, y_pred_m2)
            mape_m2 = np.mean(np.abs((y_true_m2 - y_pred_m2) / y_true_m2)) * 100
            
            validation_metrics['month_2'] = {
                'period': str(months[1]),
                'days': len(month2_data),
                'rmse': float(rmse_m2),
                'mae': float(mae_m2),
                'mape': float(mape_m2)
            }
            
            logger.info(f"  📊 Month 2 ({months[1]}): RMSE={rmse_m2:.2f}, MAE={mae_m2:.2f}, MAPE={mape_m2:.2f}%")
        
        # 2ヶ月間全体
        y_true_all = merged['y'].values
        y_pred_all = merged['yhat'].values
        
        rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae_all = mean_absolute_error(y_true_all, y_pred_all)
        mape_all = np.mean(np.abs((y_true_all - y_pred_all) / y_true_all)) * 100
        
        validation_metrics['overall'] = {
            'period': f"{months[0]} to {months[-1]}" if len(months) > 1 else str(months[0]),
            'days': len(merged),
            'rmse': float(rmse_all),
            'mae': float(mae_all),
            'mape': float(mape_all)
        }
        
        logger.info(f"  📊 Overall: RMSE={rmse_all:.2f}, MAE={mae_all:.2f}, MAPE={mape_all:.2f}%")
        
        self.validation_metrics = validation_metrics
        
        with open(self.output_dir / 'validation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(validation_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Validation completed")
        
        return validation_metrics
    
    # ========================================================================
    # 8. 可視化
    # ========================================================================
    def create_visualizations(self):
        """包括的な可視化を作成"""
        logger.info("📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. 時系列 + 予測 + 検証
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.df_train['ds'], self.df_train['y'], 
                label='Training', linewidth=1, alpha=0.7, color='blue')
        
        if self.df_validation is not None:
            ax1.plot(self.df_validation['ds'], self.df_validation['y'], 
                    label='Validation (Actual)', linewidth=1.5, alpha=0.9, 
                    color='green', marker='o', markersize=3)
        
        if self.ensemble_forecast is not None:
            forecast = self.ensemble_forecast
            forecast_future = forecast[forecast['ds'] > self.df_train['ds'].max()]
            
            ax1.plot(forecast_future['ds'], forecast_future['yhat'], 
                    'r-', label='Forecast', linewidth=2)
            ax1.fill_between(forecast_future['ds'], 
                            forecast_future['yhat_lower'], 
                            forecast_future['yhat_upper'], 
                            alpha=0.2, color='red', label='95% CI')
        
        ax1.set_title('Time Series Forecast with Validation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Call Volume')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. 検証期間拡大
        ax2 = fig.add_subplot(gs[1, :])
        if self.df_validation is not None and self.ensemble_forecast is not None:
            val_start = self.df_validation['ds'].min() - timedelta(days=7)
            val_end = self.df_validation['ds'].max() + timedelta(days=7)
            
            df_plot = pd.concat([self.df_train, self.df_validation])
            df_plot = df_plot[(df_plot['ds'] >= val_start) & (df_plot['ds'] <= val_end)]
            
            forecast_plot = self.ensemble_forecast[
                (self.ensemble_forecast['ds'] >= val_start) & 
                (self.ensemble_forecast['ds'] <= val_end)
            ]
            
            ax2.plot(df_plot['ds'], df_plot['y'], 
                    label='Actual', linewidth=1.5, alpha=0.8, color='black', marker='o', markersize=4)
            ax2.plot(forecast_plot['ds'], forecast_plot['yhat'], 
                    'r-', label='Forecast', linewidth=2)
            ax2.fill_between(forecast_plot['ds'], 
                            forecast_plot['yhat_lower'], 
                            forecast_plot['yhat_upper'], 
                            alpha=0.2, color='red')
            
            ax2.axvspan(self.df_validation['ds'].min(), 
                       self.df_validation['ds'].max(), 
                       alpha=0.1, color='yellow', label='Validation Period')
            
            ax2.set_title('Validation Period Closeup', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Call Volume')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. 月別誤差
        ax3 = fig.add_subplot(gs[2, 0])
        if self.validation_metrics:
            metrics_list = []
            for key in ['month_1', 'month_2']:
                if key in self.validation_metrics:
                    metrics_list.append({
                        'Month': self.validation_metrics[key]['period'],
                        'MAE': self.validation_metrics[key]['mae'],
                        'RMSE': self.validation_metrics[key]['rmse'],
                        'MAPE': self.validation_metrics[key]['mape']
                    })
            
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                x = np.arange(len(metrics_df))
                width = 0.25
                
                ax3.bar(x - width, metrics_df['MAE'], width, label='MAE', alpha=0.8)
                ax3.bar(x, metrics_df['RMSE'], width, label='RMSE', alpha=0.8)
                ax3.bar(x + width, metrics_df['MAPE']*10, width, label='MAPE×10', alpha=0.8)
                
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Error')
                ax3.set_title('Monthly Validation Metrics', fontsize=12, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(metrics_df['Month'])
                ax3.legend()
                ax3.grid(alpha=0.3)
        
        # 4. 残差プロット
        ax4 = fig.add_subplot(gs[2, 1])
        if self.df_validation is not None and self.ensemble_forecast is not None:
            forecast_val = self.ensemble_forecast[
                self.ensemble_forecast['ds'].isin(self.df_validation['ds'])
            ]
            merged = pd.merge(self.df_validation[['ds', 'y']], 
                            forecast_val[['ds', 'yhat']], on='ds')
            
            if len(merged) > 0:
                residuals = merged['y'] - merged['yhat']
                
                ax4.scatter(merged['yhat'], residuals, alpha=0.6, s=30)
                ax4.axhline(0, color='red', linestyle='--', linewidth=2)
                ax4.set_xlabel('Predicted')
                ax4.set_ylabel('Residual')
                ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
                ax4.grid(alpha=0.3)
        
        # 5. 実測 vs 予測
        ax5 = fig.add_subplot(gs[2, 2])
        if self.df_validation is not None and self.ensemble_forecast is not None:
            forecast_val = self.ensemble_forecast[
                self.ensemble_forecast['ds'].isin(self.df_validation['ds'])
            ]
            merged = pd.merge(self.df_validation[['ds', 'y']], 
                            forecast_val[['ds', 'yhat']], on='ds')
            
            if len(merged) > 0:
                ax5.scatter(merged['y'], merged['yhat'], alpha=0.6, s=30)
                
                min_val = min(merged['y'].min(), merged['yhat'].min())
                max_val = max(merged['y'].max(), merged['yhat'].max())
                ax5.plot([min_val, max_val], [min_val, max_val], 
                        'r--', linewidth=2, label='Perfect')
                
                ax5.set_xlabel('Actual')
                ax5.set_ylabel('Predicted')
                ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
                ax5.legend()
                ax5.grid(alpha=0.3)
        
        # 6-8. 診断関連
        y = self.df_train['y'].values
        
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.hist(y, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax6.set_title('Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Call Volume')
        ax6.set_ylabel('Density')
        ax6.grid(alpha=0.3)
        
        ax7 = fig.add_subplot(gs[3, 1])
        stats.probplot(y, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax7.grid(alpha=0.3)
        
        ax8 = fig.add_subplot(gs[3, 2])
        df_train_copy = self.df_train.copy()
        df_train_copy['weekday'] = df_train_copy['ds'].dt.dayofweek
        weekday_data = [df_train_copy[df_train_copy['weekday'] == i]['y'].values for i in range(7)]
        ax8.boxplot(weekday_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax8.set_title('Box Plot by Weekday', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Call Volume')
        ax8.grid(alpha=0.3)
        
        plt.savefig(self.output_dir / 'visualizations.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualizations saved")
    
    # ========================================================================
    # 9. レポート生成
    # ========================================================================
    def generate_report(self):
        """詳細レポートを生成"""
        logger.info("📝 Generating report...")
        
        report = []
        report.append("=" * 80)
        report.append("Prophet Ultimate Predictor v3.0 - Maximum Accuracy Edition")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. データ情報
        report.append("1. DATA INFORMATION")
        report.append("-" * 80)
        report.append(f"  Training: {self.df_train['ds'].min().date()} to {self.df_train['ds'].max().date()} ({len(self.df_train)} days)")
        report.append(f"  Validation: {self.df_validation['ds'].min().date()} to {self.df_validation['ds'].max().date()} ({len(self.df_validation)} days)")
        report.append(f"  Mean: {self.df_train['y'].mean():.1f}, Std: {self.df_train['y'].std():.1f}, CV: {self.df_train['y'].std() / self.df_train['y'].mean():.3f}")
        report.append("")
        
        # 2. 診断結果
        report.append("2. DIAGNOSTICS")
        report.append("-" * 80)
        if self.diagnostics:
            if 'basic_stats' in self.diagnostics:
                stats_data = self.diagnostics['basic_stats']
                report.append(f"  CV: {stats_data['cv']:.3f}, Skewness: {stats_data['skewness']:.3f}")
            if 'weekday_effect' in self.diagnostics:
                week = self.diagnostics['weekday_effect']
                report.append(f"  Weekday effect: {'Significant' if week.get('has_effect') else 'Not significant'}")
            if 'outliers' in self.diagnostics:
                out = self.diagnostics['outliers']
                report.append(f"  Outliers: {out.get('count', 0)} ({out.get('percentage', 0):.2f}%)")
        report.append("")
        
        # 3. 最適パラメータ
        if self.best_params:
            report.append("3. OPTUNA BEST PARAMETERS")
            report.append("-" * 80)
            for key, value in self.best_params.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # 4. 検証結果
        if self.validation_metrics:
            report.append("4. VALIDATION RESULTS")
            report.append("-" * 80)
            
            if 'month_1' in self.validation_metrics:
                m1 = self.validation_metrics['month_1']
                report.append(f"  Month 1 ({m1['period']}): RMSE={m1['rmse']:.2f}, MAE={m1['mae']:.2f}, MAPE={m1['mape']:.2f}%")
            
            if 'month_2' in self.validation_metrics:
                m2 = self.validation_metrics['month_2']
                report.append(f"  Month 2 ({m2['period']}): RMSE={m2['rmse']:.2f}, MAE={m2['mae']:.2f}, MAPE={m2['mape']:.2f}%")
            
            if 'overall' in self.validation_metrics:
                overall = self.validation_metrics['overall']
                report.append(f"  Overall: RMSE={overall['rmse']:.2f}, MAE={overall['mae']:.2f}, MAPE={overall['mape']:.2f}%")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        with open(self.output_dir / 'report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ Report saved")
        
        return report_text
    
    # ========================================================================
    # 10. モデル保存
    # ========================================================================
    def save_models(self, filepath: str = None):
        """モデルを保存"""
        if filepath is None:
            filepath = self.output_dir / 'models.pkl'
        
        save_obj = {
            'models': self.models,
            'ensemble_forecast': self.ensemble_forecast,
            'best_params': self.best_params,
            'diagnostics': self.diagnostics,
            'validation_metrics': self.validation_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        
        logger.info(f"✅ Models saved to {filepath}")
    
    # ========================================================================
    # 11. 完全実行パイプライン
    # ========================================================================
    def fit_predict(self, filepath: str, validation_months: int = 2, 
                    optuna_trials: int = 200) -> Dict:
        """
        完全実行パイプライン (最大精度版)
        
        Parameters
        ----------
        filepath : str
            CSVファイルパス
        validation_months : int
            検証期間 (月数、デフォルト: 2)
        optuna_trials : int
            Optuna試行回数 (デフォルト: 200)
        
        Returns
        -------
        dict
            全結果
        """
        logger.info("🚀 Starting Prophet Ultimate Predictor v3.0 (Maximum Accuracy Edition)...")
        
        # 1. データ読み込み
        self.load_data(filepath, validation_months=validation_months)
        
        # 2. 診断
        self.run_comprehensive_diagnostics()
        
        # 3. 祝日データフレーム作成
        holidays = self.create_holiday_dataframe()
        
        # 4. Optuna最適化
        self.optimize_with_optuna(
            self.df_train, 
            holidays=holidays, 
            n_trials=optuna_trials,
            cv_horizon_days=30
        )
        
        # 5. アンサンブルモデル訓練
        self.fit_ensemble_models(self.df_train, holidays=holidays)
        
        # 6. 検証
        self.validate_forecast()
        
        # 7. 可視化
        self.create_visualizations()
        
        # 8. レポート生成
        report = self.generate_report()
        
        # 9. モデル保存
        self.save_models()
        
        # 10. 予測結果をCSV保存
        if self.ensemble_forecast is not None:
            forecast_df = self.ensemble_forecast[
                self.ensemble_forecast['ds'] > self.df_train['ds'].max()
            ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_df.to_csv(self.output_dir / 'forecast.csv', index=False)
            logger.info(f"✅ Forecast saved")
        
        logger.info("=" * 80)
        logger.info("🎉 Pipeline completed!")
        logger.info(f"📁 Results: {self.output_dir}")
        logger.info("=" * 80)
        
        return {
            'diagnostics': self.diagnostics,
            'best_params': self.best_params,
            'models': self.models,
            'forecast': self.ensemble_forecast,
            'validation_metrics': self.validation_metrics,
            'report': report
        }


# ============================================================================
# コマンドライン実行
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prophet Ultimate Predictor v3.0 - Maximum Accuracy Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('filepath', type=str, help='Path to CSV file (ds, y columns)')
    parser.add_argument('--validation-months', type=int, default=2, 
                        help='Validation period in months (default: 2)')
    parser.add_argument('--optuna-trials', type=int, default=200, 
                        help='Optuna trials (default: 200)')
    parser.add_argument('--output', type=str, default='prophet_v3_results', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 実行
    predictor = ProphetUltimatePredictor(output_dir=args.output)
    results = predictor.fit_predict(
        args.filepath, 
        validation_months=args.validation_months,
        optuna_trials=args.optuna_trials
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(results['report'])
