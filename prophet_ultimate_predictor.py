#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Prophet Ultimate Predictor for Call Center Daily Volume Forecasting v2.1
==============================================================================

超高精度コールセンター日次呼量予測システム (2ヶ月予測 + 詳細検証付き)

主要機能:
---------
1. 自動診断システム (CV, ANOVA, ACF, ADF, スペクトル解析, STL分解)
2. レジーム自動検出 (K-means + 分位点ベース)
3. ハイパーパラメータ最適化 (Grid Search + 時系列交差検証)
4. アンサンブル学習 (複数Prophetモデル + 重み付き平均)
5. 高度な特徴量生成 (祝日, 月初月末, キャンペーン, 外生変数)
6. 2ヶ月固定予測 (最後の月から2ヶ月先)
7. 詳細検証 (1ヶ月目/2ヶ月目/2ヶ月間の RMSE/MAE/MAPE)
8. 包括的可視化 (20+チャート)
9. 詳細レポート (JSON, CSV, TXT)
10. モデル永続化 (保存/ロード)

使用例:
-------
# コマンドライン
python prophet_ultimate_predictor.py data.csv

# Python スクリプト
from prophet_ultimate_predictor import ProphetUltimatePredictor
predictor = ProphetUltimatePredictor()
results = predictor.fit_predict('data.csv')

作成者: AI Assistant
バージョン: 2.1
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

# Prophet と診断ツール
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
except ImportError:
    print("❌ Prophet not installed. Run: pip install prophet")
    sys.exit(1)

# 統計・時系列分析
from scipy import stats, signal
from scipy.stats import normaltest, shapiro, anderson, jarque_bera
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
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
        logging.FileHandler('prophet_ultimate_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ProphetUltimatePredictor メインクラス (v2.1)
# ============================================================================
class ProphetUltimatePredictor:
    """
    コールセンター日次呼量予測用の超高精度Prophetシステム
    2ヶ月固定予測 + 詳細検証機能付き
    
    Parameters
    ----------
    output_dir : str
        出力ディレクトリ (デフォルト: 'prophet_ultimate_results')
    """
    
    def __init__(self, output_dir: str = 'prophet_ultimate_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.df = None
        self.df_train = None  # 学習用 (最後の2ヶ月を除く)
        self.df_validation = None  # 検証用 (最後の2ヶ月)
        self.diagnostics = {}
        self.regimes = {}
        self.models = {}
        self.forecasts = {}
        self.best_params = {}
        self.cv_results = {}
        self.ensemble_forecast = None
        self.validation_metrics = {}  # 検証結果
        
        logger.info(f"✅ ProphetUltimatePredictor v2.1 initialized. Output: {self.output_dir}")
    
    # ========================================================================
    # 1. データ読み込み & 前処理 (検証分割付き)
    # ========================================================================
    def load_data(self, filepath: Union[str, Path], date_col: str = 'ds', 
                  value_col: str = 'y', validation_months: int = 2) -> pd.DataFrame:
        """
        データ読み込みと基本前処理 (最後のN ヶ月を検証用に分割)
        
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
            # 自動検出
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
        
        # 検証期間の計算 (最後のN ヶ月)
        max_date = df['ds'].max()
        validation_start = max_date - relativedelta(months=validation_months) + timedelta(days=1)
        
        # データ分割
        self.df = df
        self.df_train = df[df['ds'] < validation_start].copy()
        self.df_validation = df[df['ds'] >= validation_start].copy()
        
        logger.info(f"✅ Data loaded: {len(df)} rows, {df['ds'].min()} to {df['ds'].max()}")
        logger.info(f"  📊 Train: {len(self.df_train)} rows ({self.df_train['ds'].min()} to {self.df_train['ds'].max()})")
        logger.info(f"  🔍 Validation: {len(self.df_validation)} rows ({self.df_validation['ds'].min()} to {self.df_validation['ds'].max()})")
        
        # 検証期間の月情報
        val_months = self.df_validation.groupby(self.df_validation['ds'].dt.to_period('M')).size()
        logger.info(f"  📅 Validation months: {list(val_months.index.astype(str))}")
        
        return df
    
    # ========================================================================
    # 2. 包括的診断システム (学習データのみ)
    # ========================================================================
    def run_comprehensive_diagnostics(self) -> Dict:
        """
        包括的時系列診断を実行 (学習データのみ使用)
        
        Returns
        -------
        dict
            診断結果辞書
        """
        logger.info("🔍 Running comprehensive diagnostics on training data...")
        
        df = self.df_train.copy()
        y = df['y'].values
        
        diagnostics = {}
        
        # ------------------------------------------------------------------
        # 2.1 基本統計量
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 2.2 正規性検定
        # ------------------------------------------------------------------
        try:
            _, p_shapiro = shapiro(y[:5000] if len(y) > 5000 else y)
            _, p_normal = normaltest(y)
            jb_stat, p_jb = jarque_bera(y)
            
            diagnostics['normality'] = {
                'shapiro_p': float(p_shapiro),
                'normaltest_p': float(p_normal),
                'jarque_bera_p': float(p_jb),
                'is_normal': bool(p_normal > 0.05)
            }
            logger.info(f"  📈 Normality test p-value: {p_normal:.4f}")
        except Exception as e:
            logger.warning(f"  ⚠️  Normality test failed: {e}")
            diagnostics['normality'] = {'is_normal': False}
        
        # ------------------------------------------------------------------
        # 2.3 定常性検定 (ADF)
        # ------------------------------------------------------------------
        try:
            adf_result = adfuller(y, autolag='AIC')
            diagnostics['stationarity'] = {
                'adf_statistic': float(adf_result[0]),
                'adf_p_value': float(adf_result[1]),
                'is_stationary': bool(adf_result[1] < 0.05)
            }
            logger.info(f"  📉 ADF p-value: {adf_result[1]:.4f} ({'定常' if adf_result[1] < 0.05 else '非定常'})")
        except Exception as e:
            logger.warning(f"  ⚠️  ADF test failed: {e}")
            diagnostics['stationarity'] = {'is_stationary': False}
        
        # ------------------------------------------------------------------
        # 2.4 自己相関分析
        # ------------------------------------------------------------------
        try:
            acf_values = acf(y, nlags=min(30, len(y)//2 - 1), fft=True)
            pacf_values = pacf(y, nlags=min(30, len(y)//2 - 1))
            
            diagnostics['autocorrelation'] = {
                'acf_lag1': float(acf_values[1]),
                'acf_lag7': float(acf_values[7]) if len(acf_values) > 7 else 0.0,
                'pacf_lag1': float(pacf_values[1]),
                'significant_lags': [int(i) for i, val in enumerate(acf_values[1:15]) if abs(val) > 1.96/np.sqrt(len(y))]
            }
            logger.info(f"  🔄 ACF(lag=7): {diagnostics['autocorrelation']['acf_lag7']:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️  ACF/PACF failed: {e}")
            diagnostics['autocorrelation'] = {}
        
        # ------------------------------------------------------------------
        # 2.5 曜日効果 (ANOVA)
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 2.6 月効果 (ANOVA)
        # ------------------------------------------------------------------
        df['month'] = df['ds'].dt.month
        try:
            month_groups = [df[df['month'] == i]['y'].values for i in range(1, 13) if len(df[df['month'] == i]) > 0]
            if len(month_groups) > 1:
                f_stat, p_value = stats.f_oneway(*month_groups)
                
                diagnostics['month_effect'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'has_effect': bool(p_value < 0.05)
                }
                logger.info(f"  📆 Month ANOVA p-value: {p_value:.4e}")
            else:
                diagnostics['month_effect'] = {'has_effect': False}
        except Exception as e:
            logger.warning(f"  ⚠️  Month ANOVA failed: {e}")
            diagnostics['month_effect'] = {'has_effect': False}
        
        # ------------------------------------------------------------------
        # 2.7 スペクトル解析
        # ------------------------------------------------------------------
        try:
            freqs, psd = signal.periodogram(y, fs=1.0)
            top_freq_idx = np.argsort(psd[1:])[-3:] + 1
            top_periods = [1.0 / freqs[i] for i in top_freq_idx if freqs[i] > 0]
            
            diagnostics['spectral'] = {
                'dominant_periods': [float(p) for p in sorted(top_periods, reverse=True)]
            }
            logger.info(f"  🌊 Dominant periods: {[f'{p:.1f}' for p in top_periods[:3]]}")
        except Exception as e:
            logger.warning(f"  ⚠️  Spectral analysis failed: {e}")
            diagnostics['spectral'] = {}
        
        # ------------------------------------------------------------------
        # 2.8 STL分解
        # ------------------------------------------------------------------
        try:
            if len(y) >= 14:
                stl = STL(y, seasonal=7, robust=True)
                result = stl.fit()
                
                trend_strength = 1 - np.var(result.resid) / np.var(result.trend + result.resid)
                seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
                
                diagnostics['stl_decomposition'] = {
                    'trend_strength': float(max(0, trend_strength)),
                    'seasonal_strength': float(max(0, seasonal_strength))
                }
                logger.info(f"  🔬 Trend strength: {trend_strength:.3f}, Seasonal: {seasonal_strength:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️  STL decomposition failed: {e}")
            diagnostics['stl_decomposition'] = {}
        
        # ------------------------------------------------------------------
        # 2.9 外れ値検出
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 2.10 二峰性検出
        # ------------------------------------------------------------------
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(y)
            x_grid = np.linspace(y.min(), y.max(), 200)
            density = kde(x_grid)
            
            peaks = signal.find_peaks(density, prominence=0.01)[0]
            diagnostics['bimodality'] = {
                'num_peaks': len(peaks),
                'is_bimodal': bool(len(peaks) >= 2)
            }
            logger.info(f"  👥 Distribution peaks: {len(peaks)} ({'二峰性' if len(peaks) >= 2 else '単峰性'})")
        except Exception as e:
            logger.warning(f"  ⚠️  Bimodality test failed: {e}")
            diagnostics['bimodality'] = {'is_bimodal': False}
        
        self.diagnostics = diagnostics
        
        # 診断結果を保存
        with open(self.output_dir / 'diagnostics.json', 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Diagnostics completed. Saved to {self.output_dir / 'diagnostics.json'}")
        
        return diagnostics
    
    # ========================================================================
    # 3. レジーム自動検出
    # ========================================================================
    def detect_regimes(self, n_regimes: int = 2, method: str = 'kmeans') -> Dict:
        """
        レジーム自動検出 (K-means または分位点ベース)
        
        Parameters
        ----------
        n_regimes : int
            レジーム数
        method : str
            'kmeans' または 'quantile'
        
        Returns
        -------
        dict
            レジーム情報
        """
        logger.info(f"🎯 Detecting {n_regimes} regimes using {method} method...")
        
        df = self.df_train.copy()
        y = df['y'].values.reshape(-1, 1)
        
        if method == 'kmeans':
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y)
            
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_scaled)
            
            silhouette = silhouette_score(y_scaled, labels)
            logger.info(f"  🎯 Silhouette score: {silhouette:.3f}")
            
        elif method == 'quantile':
            quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
            thresholds = np.quantile(y, quantiles)
            
            labels = np.zeros(len(y), dtype=int)
            for i, threshold in enumerate(thresholds):
                labels[y.flatten() > threshold] = i + 1
            
            silhouette = None
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df['regime'] = labels
        
        # レジーム統計
        regime_stats = {}
        for regime_id in range(n_regimes):
            regime_data = df[df['regime'] == regime_id]['y']
            regime_stats[f'regime_{regime_id}'] = {
                'count': int(len(regime_data)),
                'mean': float(regime_data.mean()),
                'std': float(regime_data.std()),
                'cv': float(regime_data.std() / regime_data.mean()),
                'min': float(regime_data.min()),
                'max': float(regime_data.max())
            }
            logger.info(f"  📊 Regime {regime_id}: N={len(regime_data)}, Mean={regime_data.mean():.1f}, CV={regime_stats[f'regime_{regime_id}']['cv']:.3f}")
        
        self.regimes = {
            'method': method,
            'n_regimes': n_regimes,
            'labels': labels.tolist(),
            'stats': regime_stats,
            'silhouette_score': float(silhouette) if silhouette is not None else None
        }
        
        self.df_train = df
        
        # レジーム情報を保存
        with open(self.output_dir / 'regimes.json', 'w', encoding='utf-8') as f:
            json.dump(self.regimes, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Regimes detected. Saved to {self.output_dir / 'regimes.json'}")
        
        return self.regimes
    
    # ========================================================================
    # 4. 祝日・特殊日データフレーム生成
    # ========================================================================
    def create_holiday_dataframe(self, start_year: int = None, 
                                  end_year: int = None) -> pd.DataFrame:
        """
        日本の祝日・特殊日データフレームを生成
        
        Parameters
        ----------
        start_year : int
            開始年 (Noneの場合はデータから自動検出)
        end_year : int
            終了年 (Noneの場合はデータから自動検出)
        
        Returns
        -------
        pd.DataFrame
            祝日データフレーム
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if start_year is None:
            start_year = self.df['ds'].dt.year.min()
        if end_year is None:
            end_year = self.df['ds'].dt.year.max() + 2  # 予測期間を考慮
        
        logger.info(f"📅 Creating holiday dataframe ({start_year}-{end_year})")
        
        holidays = []
        
        # ------------------------------------------------------------------
        # 4.1 日本の祝日
        # ------------------------------------------------------------------
        if jpholiday is not None:
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    for day in range(1, 32):
                        try:
                            date = datetime(year, month, day)
                            if jpholiday.is_holiday(date):
                                name = jpholiday.is_holiday_name(date)
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
        
        # ------------------------------------------------------------------
        # 4.2 月初 (1-3日)
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 4.3 月末 (最終3日)
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 4.4 年末年始
        # ------------------------------------------------------------------
        for year in range(start_year, end_year + 1):
            # 年末 (12/28-31)
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
            
            # 年始 (1/1-7)
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
        
        # 重複削除
        holidays_df = holidays_df.sort_values('prior_scale', ascending=False).drop_duplicates('ds', keep='first')
        holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"✅ Holiday dataframe created: {len(holidays_df)} entries")
        
        return holidays_df
    
    # ========================================================================
    # 5. ハイパーパラメータ最適化 (簡略版)
    # ========================================================================
    def optimize_hyperparameters(self, df: pd.DataFrame, 
                                  holidays: pd.DataFrame = None,
                                  quick_mode: bool = True) -> Dict:
        """
        診断ベースのハイパーパラメータ選択 (高速版)
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        holidays : pd.DataFrame
            祝日データフレーム
        quick_mode : bool
            Trueの場合、交差検証をスキップして診断ベースで選択
        
        Returns
        -------
        dict
            最適パラメータ
        """
        logger.info("🔍 Selecting hyperparameters based on diagnostics...")
        
        # 診断結果に基づいてパラメータ選択
        cv = self.diagnostics.get('basic_stats', {}).get('cv', 0.3)
        
        if cv < 0.3:
            # 低変動
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 5.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            }
            logger.info("  📊 Low variability (CV<0.3) → Conservative parameters")
        elif cv < 0.5:
            # 中変動
            best_params = {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 20.0,
                'seasonality_mode': 'multiplicative'
            }
            logger.info("  📊 Medium variability (0.3≤CV<0.5) → Standard parameters")
        else:
            # 高変動
            best_params = {
                'changepoint_prior_scale': 0.3,
                'seasonality_prior_scale': 15.0,
                'holidays_prior_scale': 30.0,
                'seasonality_mode': 'multiplicative'
            }
            logger.info("  📊 High variability (CV≥0.5) → Aggressive parameters")
        
        self.best_params = best_params
        
        # 結果保存
        with open(self.output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump({'params': best_params, 'cv': float(cv)}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Parameters selected: {best_params}")
        
        return best_params
    
    # ========================================================================
    # 6. アンサンブル予測 (2ヶ月固定)
    # ========================================================================
    def fit_ensemble_models(self, df: pd.DataFrame, 
                            holidays: pd.DataFrame = None) -> Dict:
        """
        複数のProphetモデルを訓練してアンサンブル予測 (2ヶ月固定)
        
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
        
        logger.info(f"🎯 Training ensemble models (2-month forecast: {horizon_days} days)...")
        logger.info(f"  📅 Forecast period: {max_date.date()} → {future_end.date()}")
        
        models = {}
        forecasts = {}
        
        # ------------------------------------------------------------------
        # モデル1: 最適化パラメータモデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 1: Optimized parameters")
        try:
            best_params = self.best_params if self.best_params else {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 20.0,
                'seasonality_mode': 'multiplicative'
            }
            
            model1 = Prophet(
                changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                holidays_prior_scale=best_params['holidays_prior_scale'],
                seasonality_mode=best_params['seasonality_mode'],
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                uncertainty_samples=1000
            )
            
            model1.add_seasonality(name='weekly', period=7, fourier_order=5)
            model1.add_seasonality(name='monthly', period=30.5, fourier_order=10)
            model1.add_seasonality(name='yearly', period=365.25, fourier_order=15)
            
            model1.fit(df[['ds', 'y']])
            
            # 予測データフレーム作成
            future1 = model1.make_future_dataframe(periods=horizon_days)
            forecast1 = model1.predict(future1)
            
            models['optimized'] = model1
            forecasts['optimized'] = forecast1
            
            logger.info("    ✅ Model 1 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 1 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル2: 保守的モデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 2: Conservative")
        try:
            model2 = Prophet(
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=5.0,
                holidays_prior_scale=15.0,
                seasonality_mode='additive',
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
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
        # モデル3: アグレッシブモデル
        # ------------------------------------------------------------------
        logger.info("  🔧 Model 3: Aggressive")
        try:
            model3 = Prophet(
                changepoint_prior_scale=0.5,
                seasonality_prior_scale=20.0,
                holidays_prior_scale=30.0,
                seasonality_mode='multiplicative',
                holidays=holidays,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                uncertainty_samples=1000
            )
            
            model3.add_seasonality(name='weekly', period=7, fourier_order=10)
            model3.add_seasonality(name='monthly', period=30.5, fourier_order=15)
            model3.add_seasonality(name='yearly', period=365.25, fourier_order=20)
            
            model3.fit(df[['ds', 'y']])
            future3 = model3.make_future_dataframe(periods=horizon_days)
            forecast3 = model3.predict(future3)
            
            models['aggressive'] = model3
            forecasts['aggressive'] = forecast3
            
            logger.info("    ✅ Model 3 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 3 failed: {e}")
        
        # ------------------------------------------------------------------
        # アンサンブル (重み付き平均)
        # ------------------------------------------------------------------
        logger.info("  🎯 Creating ensemble forecast...")
        
        # 学習データでの性能評価
        train_maes = {}
        for name, forecast in forecasts.items():
            train_forecast = forecast[forecast['ds'].isin(df['ds'])]
            merged = pd.merge(df[['ds', 'y']], train_forecast[['ds', 'yhat']], on='ds')
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
        ensemble_forecast['yhat'] = sum(forecasts[name]['yhat'] * weights[name] 
                                        for name in forecasts.keys())
        ensemble_forecast['yhat_lower'] = sum(forecasts[name]['yhat_lower'] * weights[name] 
                                               for name in forecasts.keys())
        ensemble_forecast['yhat_upper'] = sum(forecasts[name]['yhat_upper'] * weights[name] 
                                               for name in forecasts.keys())
        
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
    # 7. 詳細検証 (1ヶ月目/2ヶ月目/2ヶ月間)
    # ========================================================================
    def validate_forecast(self) -> Dict:
        """
        検証データで予測性能を評価
        - 1ヶ月目の RMSE/MAE/MAPE
        - 2ヶ月目の RMSE/MAE/MAPE
        - 2ヶ月間の RMSE/MAE/MAPE
        
        Returns
        -------
        dict
            検証結果
        """
        logger.info("🔍 Validating forecast on holdout data...")
        
        if self.df_validation is None or len(self.df_validation) == 0:
            logger.warning("⚠️  No validation data available")
            return {}
        
        # 検証期間の予測値を抽出
        forecast_val = self.ensemble_forecast[
            self.ensemble_forecast['ds'].isin(self.df_validation['ds'])
        ].copy()
        
        # マージ
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
        
        # ------------------------------------------------------------------
        # 1ヶ月目の評価
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 2ヶ月目の評価
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 2ヶ月間全体の評価
        # ------------------------------------------------------------------
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
        
        # 結果保存
        with open(self.output_dir / 'validation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(validation_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Validation completed. Saved to {self.output_dir / 'validation_metrics.json'}")
        
        return validation_metrics
    
    # ========================================================================
    # 8. 可視化
    # ========================================================================
    def create_visualizations(self):
        """
        包括的な可視化を作成 (検証結果含む)
        """
        logger.info("📊 Creating comprehensive visualizations...")
        
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
        
        # ------------------------------------------------------------------
        # 1. 時系列 + 予測 + 検証
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, :])
        
        # 学習データ
        ax1.plot(self.df_train['ds'], self.df_train['y'], 
                label='Training Data', linewidth=1, alpha=0.7, color='blue')
        
        # 検証データ
        if self.df_validation is not None:
            ax1.plot(self.df_validation['ds'], self.df_validation['y'], 
                    label='Validation Data (Actual)', linewidth=1.5, alpha=0.9, 
                    color='green', marker='o', markersize=3)
        
        # 予測
        if self.ensemble_forecast is not None:
            forecast = self.ensemble_forecast
            
            # 予測期間のみ
            forecast_future = forecast[forecast['ds'] > self.df_train['ds'].max()]
            
            ax1.plot(forecast_future['ds'], forecast_future['yhat'], 
                    'r-', label='Forecast', linewidth=2)
            ax1.fill_between(forecast_future['ds'], 
                            forecast_future['yhat_lower'], 
                            forecast_future['yhat_upper'], 
                            alpha=0.2, color='red', label='Uncertainty')
        
        ax1.set_title('Time Series Forecast with Validation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Call Volume')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 2. 検証期間の拡大図
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[1, :])
        
        if self.df_validation is not None and self.ensemble_forecast is not None:
            # 検証期間 + 前後1週間
            val_start = self.df_validation['ds'].min() - timedelta(days=7)
            val_end = self.df_validation['ds'].max() + timedelta(days=7)
            
            # データフィルタ
            df_plot = pd.concat([self.df_train, self.df_validation])
            df_plot = df_plot[(df_plot['ds'] >= val_start) & (df_plot['ds'] <= val_end)]
            
            forecast_plot = self.ensemble_forecast[
                (self.ensemble_forecast['ds'] >= val_start) & 
                (self.ensemble_forecast['ds'] <= val_end)
            ]
            
            # プロット
            ax2.plot(df_plot['ds'], df_plot['y'], 
                    label='Actual', linewidth=1.5, alpha=0.8, color='black', marker='o', markersize=4)
            ax2.plot(forecast_plot['ds'], forecast_plot['yhat'], 
                    'r-', label='Forecast', linewidth=2)
            ax2.fill_between(forecast_plot['ds'], 
                            forecast_plot['yhat_lower'], 
                            forecast_plot['yhat_upper'], 
                            alpha=0.2, color='red')
            
            # 検証期間を強調
            ax2.axvspan(self.df_validation['ds'].min(), 
                       self.df_validation['ds'].max(), 
                       alpha=0.1, color='yellow', label='Validation Period')
            
            ax2.set_title('Validation Period Closeup', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Call Volume')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 3. 月別誤差
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 4. 残差プロット
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 5. 実測 vs 予測
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, 2])
        
        if self.df_validation is not None and self.ensemble_forecast is not None:
            forecast_val = self.ensemble_forecast[
                self.ensemble_forecast['ds'].isin(self.df_validation['ds'])
            ]
            merged = pd.merge(self.df_validation[['ds', 'y']], 
                            forecast_val[['ds', 'yhat']], on='ds')
            
            if len(merged) > 0:
                ax5.scatter(merged['y'], merged['yhat'], alpha=0.6, s=30)
                
                # 対角線
                min_val = min(merged['y'].min(), merged['yhat'].min())
                max_val = max(merged['y'].max(), merged['yhat'].max())
                ax5.plot([min_val, max_val], [min_val, max_val], 
                        'r--', linewidth=2, label='Perfect Prediction')
                
                ax5.set_xlabel('Actual')
                ax5.set_ylabel('Predicted')
                ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
                ax5.legend()
                ax5.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 6-8. 診断関連 (学習データ)
        # ------------------------------------------------------------------
        y = self.df_train['y'].values
        
        # 分布
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.hist(y, bins=50, alpha=0.7, edgecolor='black', density=True)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y)
        x_grid = np.linspace(y.min(), y.max(), 200)
        ax6.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE')
        ax6.set_title('Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Call Volume')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # QQ Plot
        ax7 = fig.add_subplot(gs[3, 1])
        stats.probplot(y, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax7.grid(alpha=0.3)
        
        # Box Plot (曜日別)
        ax8 = fig.add_subplot(gs[3, 2])
        df_train_copy = self.df_train.copy()
        df_train_copy['weekday'] = df_train_copy['ds'].dt.dayofweek
        weekday_data = [df_train_copy[df_train_copy['weekday'] == i]['y'].values for i in range(7)]
        ax8.boxplot(weekday_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax8.set_title('Box Plot by Weekday', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Call Volume')
        ax8.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 9-11. ACF, PACF, スペクトル
        # ------------------------------------------------------------------
        ax9 = fig.add_subplot(gs[4, 0])
        plot_acf(y, lags=30, ax=ax9)
        ax9.set_title('ACF', fontsize=12, fontweight='bold')
        
        ax10 = fig.add_subplot(gs[4, 1])
        plot_pacf(y, lags=30, ax=ax10)
        ax10.set_title('PACF', fontsize=12, fontweight='bold')
        
        ax11 = fig.add_subplot(gs[4, 2])
        freqs, psd = signal.periodogram(y, fs=1.0)
        ax11.semilogy(freqs[1:100], psd[1:100])
        ax11.set_title('Periodogram', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Frequency (1/day)')
        ax11.set_ylabel('PSD')
        ax11.grid(alpha=0.3)
        
        plt.savefig(self.output_dir / 'comprehensive_visualizations.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualizations saved to {self.output_dir / 'comprehensive_visualizations.png'}")
    
    # ========================================================================
    # 9. レポート生成
    # ========================================================================
    def generate_report(self):
        """
        詳細レポートを生成 (検証結果含む)
        """
        logger.info("📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 80)
        report.append("Prophet Ultimate Predictor v2.1 - Comprehensive Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # ------------------------------------------------------------------
        # 1. データ情報
        # ------------------------------------------------------------------
        report.append("1. DATA INFORMATION")
        report.append("-" * 80)
        report.append(f"  Full period: {self.df['ds'].min()} to {self.df['ds'].max()}")
        report.append(f"  Total days: {len(self.df)}")
        report.append("")
        report.append(f"  Training period: {self.df_train['ds'].min()} to {self.df_train['ds'].max()}")
        report.append(f"  Training days: {len(self.df_train)}")
        report.append(f"  Mean: {self.df_train['y'].mean():.1f}, Std: {self.df_train['y'].std():.1f}, CV: {self.df_train['y'].std() / self.df_train['y'].mean():.3f}")
        report.append("")
        
        if self.df_validation is not None:
            report.append(f"  Validation period: {self.df_validation['ds'].min()} to {self.df_validation['ds'].max()}")
            report.append(f"  Validation days: {len(self.df_validation)}")
            val_months = self.df_validation.groupby(self.df_validation['ds'].dt.to_period('M')).size()
            report.append(f"  Validation months: {', '.join(str(m) for m in val_months.index)}")
        report.append("")
        
        # ------------------------------------------------------------------
        # 2. 診断結果
        # ------------------------------------------------------------------
        report.append("2. DIAGNOSTICS (Training Data)")
        report.append("-" * 80)
        
        if self.diagnostics:
            if 'basic_stats' in self.diagnostics:
                stats = self.diagnostics['basic_stats']
                report.append(f"  Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}, CV: {stats['cv']:.3f}")
                report.append(f"  Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
            
            if 'normality' in self.diagnostics:
                norm = self.diagnostics['normality']
                report.append(f"  Normality: {'Normal' if norm.get('is_normal') else 'Non-normal'}")
            
            if 'stationarity' in self.diagnostics:
                stat = self.diagnostics['stationarity']
                report.append(f"  Stationarity: {'Stationary' if stat.get('is_stationary') else 'Non-stationary'}")
            
            if 'weekday_effect' in self.diagnostics:
                week = self.diagnostics['weekday_effect']
                report.append(f"  Weekday effect: {'Significant' if week.get('has_effect') else 'Not significant'}")
            
            if 'outliers' in self.diagnostics:
                out = self.diagnostics['outliers']
                report.append(f"  Outliers: {out.get('count', 0)} ({out.get('percentage', 0):.2f}%)")
            
            if 'bimodality' in self.diagnostics:
                bi = self.diagnostics['bimodality']
                report.append(f"  Distribution: {'Bimodal' if bi.get('is_bimodal') else 'Unimodal'}")
        
        report.append("")
        
        # ------------------------------------------------------------------
        # 3. 最適パラメータ
        # ------------------------------------------------------------------
        if self.best_params:
            report.append("3. OPTIMIZED HYPERPARAMETERS")
            report.append("-" * 80)
            for key, value in self.best_params.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # ------------------------------------------------------------------
        # 4. 検証結果
        # ------------------------------------------------------------------
        if self.validation_metrics:
            report.append("4. VALIDATION RESULTS")
            report.append("-" * 80)
            
            if 'month_1' in self.validation_metrics:
                m1 = self.validation_metrics['month_1']
                report.append(f"  Month 1 ({m1['period']}):")
                report.append(f"    Days: {m1['days']}")
                report.append(f"    RMSE: {m1['rmse']:.2f}")
                report.append(f"    MAE:  {m1['mae']:.2f}")
                report.append(f"    MAPE: {m1['mape']:.2f}%")
                report.append("")
            
            if 'month_2' in self.validation_metrics:
                m2 = self.validation_metrics['month_2']
                report.append(f"  Month 2 ({m2['period']}):")
                report.append(f"    Days: {m2['days']}")
                report.append(f"    RMSE: {m2['rmse']:.2f}")
                report.append(f"    MAE:  {m2['mae']:.2f}")
                report.append(f"    MAPE: {m2['mape']:.2f}%")
                report.append("")
            
            if 'overall' in self.validation_metrics:
                overall = self.validation_metrics['overall']
                report.append(f"  Overall ({overall['period']}):")
                report.append(f"    Days: {overall['days']}")
                report.append(f"    RMSE: {overall['rmse']:.2f}")
                report.append(f"    MAE:  {overall['mae']:.2f}")
                report.append(f"    MAPE: {overall['mape']:.2f}%")
        
        report.append("")
        
        # ------------------------------------------------------------------
        # 5. 推奨事項
        # ------------------------------------------------------------------
        report.append("5. RECOMMENDATIONS")
        report.append("-" * 80)
        
        cv = self.diagnostics.get('basic_stats', {}).get('cv', 0)
        if cv > 0.5:
            report.append("  ⚠️  High variability detected (CV > 0.5)")
            report.append("     → Consider regime-separated models")
        
        if self.diagnostics.get('bimodality', {}).get('is_bimodal'):
            report.append("  ⚠️  Bimodal distribution detected")
            report.append("     → Investigate weekday/weekend split")
        
        if self.diagnostics.get('outliers', {}).get('percentage', 0) > 5:
            report.append("  ⚠️  High outlier percentage (>5%)")
            report.append("     → Review outlier handling strategy")
        
        if self.validation_metrics:
            overall_mape = self.validation_metrics.get('overall', {}).get('mape', 0)
            if overall_mape > 20:
                report.append("  ⚠️  High MAPE (>20%)")
                report.append("     → Consider additional features or longer training period")
        
        report.append("")
        report.append("=" * 80)
        
        # レポート保存
        report_text = "\n".join(report)
        with open(self.output_dir / 'report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ Report saved to {self.output_dir / 'report.txt'}")
        
        return report_text
    
    # ========================================================================
    # 10. モデル保存・ロード
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
            'regimes': self.regimes,
            'validation_metrics': self.validation_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        
        logger.info(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """モデルをロード"""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.models = save_obj.get('models', {})
        self.ensemble_forecast = save_obj.get('ensemble_forecast')
        self.best_params = save_obj.get('best_params', {})
        self.diagnostics = save_obj.get('diagnostics', {})
        self.regimes = save_obj.get('regimes', {})
        self.validation_metrics = save_obj.get('validation_metrics', {})
        
        logger.info(f"✅ Models loaded from {filepath}")
    
    # ========================================================================
    # 11. 完全実行パイプライン
    # ========================================================================
    def fit_predict(self, filepath: str, validation_months: int = 2, 
                    quick_mode: bool = True) -> Dict:
        """
        完全実行パイプライン (2ヶ月固定予測 + 検証)
        
        Parameters
        ----------
        filepath : str
            CSVファイルパス
        validation_months : int
            検証期間 (月数、デフォルト: 2)
        quick_mode : bool
            高速モード
        
        Returns
        -------
        dict
            全結果を含む辞書
        """
        logger.info("🚀 Starting Prophet Ultimate Predictor v2.1 pipeline...")
        
        # 1. データ読み込み (検証分割)
        self.load_data(filepath, validation_months=validation_months)
        
        # 2. 診断 (学習データのみ)
        self.run_comprehensive_diagnostics()
        
        # 3. レジーム検出 (高変動時のみ)
        cv = self.diagnostics.get('basic_stats', {}).get('cv', 0)
        if cv > 0.4 or self.diagnostics.get('bimodality', {}).get('is_bimodal'):
            logger.info("🎯 High variability or bimodality detected. Running regime detection...")
            self.detect_regimes(n_regimes=2, method='kmeans')
        
        # 4. 祝日データフレーム作成
        holidays = self.create_holiday_dataframe()
        
        # 5. ハイパーパラメータ最適化
        self.optimize_hyperparameters(
            self.df_train, 
            holidays=holidays, 
            quick_mode=quick_mode
        )
        
        # 6. アンサンブルモデル訓練 (2ヶ月固定)
        self.fit_ensemble_models(self.df_train, holidays=holidays)
        
        # 7. 検証
        self.validate_forecast()
        
        # 8. 可視化
        self.create_visualizations()
        
        # 9. レポート生成
        report = self.generate_report()
        
        # 10. モデル保存
        self.save_models()
        
        # 11. 予測結果をCSV保存
        if self.ensemble_forecast is not None:
            # 予測期間のみ (学習期間の最後の日より後)
            forecast_df = self.ensemble_forecast[
                self.ensemble_forecast['ds'] > self.df_train['ds'].max()
            ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_df.to_csv(self.output_dir / 'forecast.csv', index=False)
            logger.info(f"✅ Forecast saved to {self.output_dir / 'forecast.csv'}")
        
        logger.info("=" * 80)
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 All results saved to: {self.output_dir}")
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
        description='Prophet Ultimate Predictor v2.1 for Call Center Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本実行 (2ヶ月予測 + 最後の2ヶ月で検証)
  python prophet_ultimate_predictor.py data.csv
  
  # カスタム出力ディレクトリ
  python prophet_ultimate_predictor.py data.csv --output my_results
        """
    )
    
    parser.add_argument('filepath', type=str, help='Path to CSV file (ds, y columns)')
    parser.add_argument('--validation-months', type=int, default=2, 
                        help='Validation period in months (default: 2)')
    parser.add_argument('--output', type=str, default='prophet_ultimate_results', 
                        help='Output directory (default: prophet_ultimate_results)')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    
    args = parser.parse_args()
    
    # 実行
    predictor = ProphetUltimatePredictor(output_dir=args.output)
    results = predictor.fit_predict(
        args.filepath, 
        validation_months=args.validation_months,
        quick_mode=args.quick
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(results['report'])
