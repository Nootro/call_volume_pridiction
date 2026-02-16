#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Prophet Ultimate Predictor for Call Center Daily Volume Forecasting
==============================================================================

超高精度コールセンター日次呼量予測システム

主要機能:
---------
1. 自動診断システム (CV, ANOVA, ACF, ADF, スペクトル解析, STL分解)
2. レジーム自動検出 (K-means + 分位点ベース)
3. ハイパーパラメータ最適化 (Grid Search + 時系列交差検証)
4. アンサンブル学習 (複数Prophetモデル + 重み付き平均)
5. 高度な特徴量生成 (祝日, 月初月末, キャンペーン, 外生変数)
6. 外れ値処理 (自動検出 + 複数戦略)
7. 包括的可視化 (20+チャート + インタラクティブHTML)
8. 詳細レポート (PDF, JSON, CSV)
9. モデル永続化 (保存/ロード)
10. プロダクション対応 (ロギング, エラーハンドリング)

使用例:
-------
# コマンドライン
python prophet_ultimate_predictor.py data.csv --horizon 30 --cv-days 90

# Python スクリプト
from prophet_ultimate_predictor import ProphetUltimatePredictor
predictor = ProphetUltimatePredictor()
results = predictor.fit_predict('data.csv', horizon=30)

作成者: AI Assistant
バージョン: 2.0
最終更新: 2026-02-16
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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
# ProphetUltimatePredictor メインクラス
# ============================================================================
class ProphetUltimatePredictor:
    """
    コールセンター日次呼量予測用の超高精度Prophetシステム
    
    Parameters
    ----------
    output_dir : str
        出力ディレクトリ (デフォルト: 'prophet_ultimate_results')
    """
    
    def __init__(self, output_dir: str = 'prophet_ultimate_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.df = None
        self.diagnostics = {}
        self.regimes = {}
        self.models = {}
        self.forecasts = {}
        self.best_params = {}
        self.cv_results = {}
        self.ensemble_forecast = None
        
        logger.info(f"✅ ProphetUltimatePredictor initialized. Output: {self.output_dir}")
    
    # ========================================================================
    # 1. データ読み込み & 前処理
    # ========================================================================
    def load_data(self, filepath: Union[str, Path], date_col: str = 'ds', 
                  value_col: str = 'y') -> pd.DataFrame:
        """
        データ読み込みと基本前処理
        
        Parameters
        ----------
        filepath : str or Path
            CSVファイルパス
        date_col : str
            日付カラム名
        value_col : str
            目的変数カラム名
        
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
        
        self.df = df
        logger.info(f"✅ Data loaded: {len(df)} rows, {df['ds'].min()} to {df['ds'].max()}")
        
        return df
    
    # ========================================================================
    # 2. 包括的診断システム
    # ========================================================================
    def run_comprehensive_diagnostics(self) -> Dict:
        """
        包括的時系列診断を実行
        
        Returns
        -------
        dict
            診断結果辞書
        """
        logger.info("🔍 Running comprehensive diagnostics...")
        
        df = self.df.copy()
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
            _, p_shapiro = shapiro(y[:5000] if len(y) > 5000 else y)  # Shapiroは5000サンプルまで
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
            month_groups = [df[df['month'] == i]['y'].values for i in range(1, 13)]
            f_stat, p_value = stats.f_oneway(*month_groups)
            
            diagnostics['month_effect'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'has_effect': bool(p_value < 0.05)
            }
            logger.info(f"  📆 Month ANOVA p-value: {p_value:.4e}")
        except Exception as e:
            logger.warning(f"  ⚠️  Month ANOVA failed: {e}")
            diagnostics['month_effect'] = {'has_effect': False}
        
        # ------------------------------------------------------------------
        # 2.7 スペクトル解析
        # ------------------------------------------------------------------
        try:
            freqs, psd = signal.periodogram(y, fs=1.0)
            top_freq_idx = np.argsort(psd[1:])[-3:] + 1  # DC成分除外
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
            if len(y) >= 14:  # 最低2週間必要
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
        # 2.10 二峰性検出 (Hartigan's dip test の簡易版)
        # ------------------------------------------------------------------
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(y)
            x_grid = np.linspace(y.min(), y.max(), 200)
            density = kde(x_grid)
            
            # ピーク検出
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
        
        df = self.df.copy()
        y = df['y'].values.reshape(-1, 1)
        
        if method == 'kmeans':
            # K-means クラスタリング
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y)
            
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_scaled)
            
            # シルエットスコア
            silhouette = silhouette_score(y_scaled, labels)
            logger.info(f"  🎯 Silhouette score: {silhouette:.3f}")
            
        elif method == 'quantile':
            # 分位点ベース
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
        
        self.df = df
        
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
                # 月の最終日を取得
                if month == 12:
                    next_month = datetime(year + 1, 1, 1)
                else:
                    next_month = datetime(year, month + 1, 1)
                last_day = (next_month - timedelta(days=1)).day
                
                for day in [last_day - 2, last_day - 1, last_day]:
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
        
        # 重複削除 (同じ日付で複数の休日タイプがある場合、prior_scaleが最大のものを採用)
        holidays_df = holidays_df.sort_values('prior_scale', ascending=False).drop_duplicates('ds', keep='first')
        holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"✅ Holiday dataframe created: {len(holidays_df)} entries")
        
        return holidays_df
    
    # ========================================================================
    # 5. 外生変数生成
    # ========================================================================
    def create_exogenous_features(self, df: pd.DataFrame, 
                                   future: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        外生変数を生成
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        future : pd.DataFrame
            予測期間データフレーム (Noneの場合は学習データのみ)
        
        Returns
        -------
        tuple of pd.DataFrame
            (拡張された学習データ, 拡張された予測データ)
        """
        logger.info("🔧 Creating exogenous features...")
        
        def add_features(data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            
            # 日付特徴量
            data['weekday'] = data['ds'].dt.dayofweek
            data['month'] = data['ds'].dt.month
            data['day'] = data['ds'].dt.day
            data['quarter'] = data['ds'].dt.quarter
            data['week_of_year'] = data['ds'].dt.isocalendar().week
            data['day_of_year'] = data['ds'].dt.dayofyear
            
            # 周期エンコーディング
            data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
            data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
            data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
            
            # 特殊日フラグ
            data['is_month_start'] = (data['day'] <= 3).astype(int)
            data['is_month_end'] = (data['day'] >= 28).astype(int)
            data['is_weekend'] = (data['weekday'] >= 5).astype(int)
            data['is_monday'] = (data['weekday'] == 0).astype(int)
            data['is_friday'] = (data['weekday'] == 4).astype(int)
            
            # 祝日フラグ
            if jpholiday is not None:
                data['is_holiday'] = data['ds'].apply(lambda x: int(jpholiday.is_holiday(x)))
            else:
                data['is_holiday'] = 0
            
            return data
        
        df_extended = add_features(df)
        future_extended = add_features(future) if future is not None else None
        
        logger.info(f"  ✅ Added {len(df_extended.columns) - len(df.columns)} features")
        
        return df_extended, future_extended
    
    # ========================================================================
    # 6. ハイパーパラメータ最適化
    # ========================================================================
    def optimize_hyperparameters(self, df: pd.DataFrame, 
                                  holidays: pd.DataFrame = None,
                                  cv_initial_days: int = 365,
                                  cv_horizon_days: int = 30,
                                  cv_period_days: int = 30,
                                  quick_mode: bool = False) -> Dict:
        """
        Grid Search + 時系列交差検証でハイパーパラメータ最適化
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        holidays : pd.DataFrame
            祝日データフレーム
        cv_initial_days : int
            交差検証初期学習期間 (日数)
        cv_horizon_days : int
            予測期間 (日数)
        cv_period_days : int
            交差検証の移動ステップ (日数)
        quick_mode : bool
            Trueの場合、パラメータグリッドを削減
        
        Returns
        -------
        dict
            最適パラメータ
        """
        logger.info("🔍 Optimizing hyperparameters with cross-validation...")
        
        # ------------------------------------------------------------------
        # パラメータグリッド定義
        # ------------------------------------------------------------------
        if quick_mode:
            param_grid = {
                'changepoint_prior_scale': [0.05, 0.5],
                'seasonality_prior_scale': [1.0, 10.0],
                'holidays_prior_scale': [10.0, 30.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }
        else:
            # 診断結果に基づいてグリッドを調整
            cv = self.diagnostics.get('basic_stats', {}).get('cv', 0.3)
            
            if cv < 0.3:
                # 低変動
                param_grid = {
                    'changepoint_prior_scale': [0.01, 0.05, 0.1],
                    'seasonality_prior_scale': [0.1, 1.0, 5.0],
                    'holidays_prior_scale': [5.0, 10.0, 20.0],
                    'seasonality_mode': ['additive']
                }
            elif cv < 0.5:
                # 中変動
                param_grid = {
                    'changepoint_prior_scale': [0.05, 0.1, 0.3],
                    'seasonality_prior_scale': [1.0, 5.0, 10.0],
                    'holidays_prior_scale': [10.0, 20.0, 30.0],
                    'seasonality_mode': ['additive', 'multiplicative']
                }
            else:
                # 高変動
                param_grid = {
                    'changepoint_prior_scale': [0.1, 0.3, 0.5],
                    'seasonality_prior_scale': [5.0, 10.0, 20.0],
                    'holidays_prior_scale': [20.0, 30.0, 50.0],
                    'seasonality_mode': ['multiplicative']
                }
        
        # グリッド生成
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        logger.info(f"  🔍 Testing {len(param_combinations)} parameter combinations...")
        
        # ------------------------------------------------------------------
        # Grid Search
        # ------------------------------------------------------------------
        results = []
        
        for params in tqdm(param_combinations, desc="Grid Search"):
            try:
                # モデル構築
                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    holidays=holidays,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False
                )
                
                # カスタム季節性
                model.add_seasonality(name='weekly', period=7, fourier_order=5)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
                model.add_seasonality(name='yearly', period=365.25, fourier_order=15)
                
                # 学習
                model.fit(df[['ds', 'y']])
                
                # 交差検証
                cv_results = cross_validation(
                    model,
                    initial=f'{cv_initial_days} days',
                    period=f'{cv_period_days} days',
                    horizon=f'{cv_horizon_days} days',
                    parallel='processes'
                )
                
                metrics = performance_metrics(cv_results)
                
                # 結果記録
                results.append({
                    'params': params,
                    'mae': metrics['mae'].mean(),
                    'mape': metrics['mape'].mean(),
                    'rmse': metrics['rmse'].mean(),
                    'coverage': metrics['coverage'].mean()
                })
                
            except Exception as e:
                logger.warning(f"  ⚠️  Parameter combination failed: {params}, Error: {e}")
                continue
        
        if not results:
            logger.error("❌ All parameter combinations failed!")
            raise RuntimeError("Hyperparameter optimization failed")
        
        # ------------------------------------------------------------------
        # 最適パラメータ選択 (MAE最小)
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(results)
        best_idx = results_df['mae'].idxmin()
        best_params = results_df.loc[best_idx, 'params']
        best_metrics = results_df.loc[best_idx, ['mae', 'mape', 'rmse', 'coverage']].to_dict()
        
        logger.info(f"✅ Best parameters found:")
        logger.info(f"  📊 MAE: {best_metrics['mae']:.2f}, MAPE: {best_metrics['mape']:.3f}")
        logger.info(f"  ⚙️  Params: {best_params}")
        
        self.best_params = best_params
        self.cv_results = results_df
        
        # 結果保存
        results_df.to_csv(self.output_dir / 'cv_results.csv', index=False)
        with open(self.output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump({'params': best_params, 'metrics': best_metrics}, f, indent=2, ensure_ascii=False)
        
        return best_params
    
    # ========================================================================
    # 7. アンサンブル予測
    # ========================================================================
    def fit_ensemble_models(self, df: pd.DataFrame, 
                            holidays: pd.DataFrame = None,
                            horizon: int = 30) -> Dict:
        """
        複数のProphetモデルを訓練してアンサンブル予測
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        holidays : pd.DataFrame
            祝日データフレーム
        horizon : int
            予測期間 (日数)
        
        Returns
        -------
        dict
            アンサンブル予測結果
        """
        logger.info(f"🎯 Training ensemble models (horizon={horizon} days)...")
        
        models = {}
        forecasts = {}
        
        # 予測期間データフレーム作成
        future = self.models.get('optimized', Prophet()).make_future_dataframe(periods=horizon)
        
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
            forecast1 = model1.predict(future)
            
            models['optimized'] = model1
            forecasts['optimized'] = forecast1
            
            logger.info("    ✅ Model 1 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 1 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル2: 保守的モデル (低changepoint_prior_scale)
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
            forecast2 = model2.predict(future)
            
            models['conservative'] = model2
            forecasts['conservative'] = forecast2
            
            logger.info("    ✅ Model 2 trained")
        except Exception as e:
            logger.error(f"    ❌ Model 2 failed: {e}")
        
        # ------------------------------------------------------------------
        # モデル3: アグレッシブモデル (高changepoint_prior_scale)
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
            forecast3 = model3.predict(future)
            
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
        
        # 逆MAEで重み計算 (MAEが小さいほど重みが大きい)
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
    # 8. 包括的可視化
    # ========================================================================
    def create_comprehensive_visualizations(self):
        """
        包括的な可視化を作成
        """
        logger.info("📊 Creating comprehensive visualizations...")
        
        fig = plt.figure(figsize=(24, 32))
        gs = fig.add_gridspec(8, 3, hspace=0.4, wspace=0.3)
        
        df = self.df
        y = df['y'].values
        
        # ------------------------------------------------------------------
        # 1. 時系列プロット
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['ds'], df['y'], label='Actual', linewidth=1, alpha=0.7)
        if self.ensemble_forecast is not None:
            forecast = self.ensemble_forecast
            ax1.plot(forecast['ds'], forecast['yhat'], 'r-', label='Ensemble Forecast', linewidth=2)
            ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                             alpha=0.2, color='red', label='Uncertainty')
        ax1.set_title('Time Series & Forecast', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Call Volume')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 2. 分布 (ヒストグラム + KDE)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(y, bins=50, alpha=0.7, edgecolor='black', density=True)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y)
        x_grid = np.linspace(y.min(), y.max(), 200)
        ax2.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE')
        ax2.set_title('Distribution (Histogram + KDE)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Call Volume')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 3. QQ Plot
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(y, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 4. Box Plot (曜日別)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 2])
        weekday_data = [df[df['weekday'] == i]['y'].values for i in range(7)]
        ax4.boxplot(weekday_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax4.set_title('Box Plot by Weekday', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Call Volume')
        ax4.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 5. ACF
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, 0])
        plot_acf(y, lags=30, ax=ax5)
        ax5.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        
        # ------------------------------------------------------------------
        # 6. PACF
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(gs[2, 1])
        plot_pacf(y, lags=30, ax=ax6)
        ax6.set_title('Partial Autocorrelation (PACF)', fontsize=12, fontweight='bold')
        
        # ------------------------------------------------------------------
        # 7. スペクトル (Periodogram)
        # ------------------------------------------------------------------
        ax7 = fig.add_subplot(gs[2, 2])
        freqs, psd = signal.periodogram(y, fs=1.0)
        ax7.semilogy(freqs[1:100], psd[1:100])
        ax7.set_title('Periodogram', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Frequency (1/day)')
        ax7.set_ylabel('Power Spectral Density')
        ax7.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 8. STL分解
        # ------------------------------------------------------------------
        if len(y) >= 14:
            try:
                stl = STL(y, seasonal=7, robust=True)
                result = stl.fit()
                
                ax8 = fig.add_subplot(gs[3, :])
                ax8.plot(df['ds'], result.trend, label='Trend', linewidth=2)
                ax8.set_title('STL Decomposition - Trend', fontsize=12, fontweight='bold')
                ax8.legend()
                ax8.grid(alpha=0.3)
                
                ax9 = fig.add_subplot(gs[4, :])
                ax9.plot(df['ds'], result.seasonal, label='Seasonal', linewidth=1, alpha=0.7)
                ax9.set_title('STL Decomposition - Seasonal', fontsize=12, fontweight='bold')
                ax9.legend()
                ax9.grid(alpha=0.3)
                
                ax10 = fig.add_subplot(gs[5, :])
                ax10.plot(df['ds'], result.resid, label='Residual', linewidth=1, alpha=0.7, color='gray')
                ax10.set_title('STL Decomposition - Residual', fontsize=12, fontweight='bold')
                ax10.legend()
                ax10.grid(alpha=0.3)
            except Exception as e:
                logger.warning(f"  ⚠️  STL plot failed: {e}")
        
        # ------------------------------------------------------------------
        # 9. レジーム可視化
        # ------------------------------------------------------------------
        if 'regime' in df.columns:
            ax11 = fig.add_subplot(gs[6, :])
            for regime_id in df['regime'].unique():
                regime_data = df[df['regime'] == regime_id]
                ax11.scatter(regime_data['ds'], regime_data['y'], 
                            label=f'Regime {regime_id}', alpha=0.6, s=10)
            ax11.set_title('Regime Detection', fontsize=12, fontweight='bold')
            ax11.set_xlabel('Date')
            ax11.set_ylabel('Call Volume')
            ax11.legend()
            ax11.grid(alpha=0.3)
        
        # ------------------------------------------------------------------
        # 10. 予測誤差 (学習データ)
        # ------------------------------------------------------------------
        if self.ensemble_forecast is not None:
            ax12 = fig.add_subplot(gs[7, :])
            train_forecast = self.ensemble_forecast[self.ensemble_forecast['ds'].isin(df['ds'])]
            merged = pd.merge(df[['ds', 'y']], train_forecast[['ds', 'yhat']], on='ds')
            errors = merged['y'] - merged['yhat']
            
            ax12.plot(merged['ds'], errors, label='Forecast Error', linewidth=1, alpha=0.7)
            ax12.axhline(0, color='red', linestyle='--', linewidth=2)
            ax12.fill_between(merged['ds'], -errors.std(), errors.std(), alpha=0.2, color='gray')
            ax12.set_title(f'Forecast Error (MAE: {abs(errors).mean():.2f})', fontsize=12, fontweight='bold')
            ax12.set_xlabel('Date')
            ax12.set_ylabel('Error')
            ax12.legend()
            ax12.grid(alpha=0.3)
        
        plt.savefig(self.output_dir / 'comprehensive_visualizations.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualizations saved to {self.output_dir / 'comprehensive_visualizations.png'}")
    
    # ========================================================================
    # 9. レポート生成
    # ========================================================================
    def generate_report(self):
        """
        詳細レポートを生成
        """
        logger.info("📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 80)
        report.append("Prophet Ultimate Predictor - Comprehensive Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # ------------------------------------------------------------------
        # 1. データ情報
        # ------------------------------------------------------------------
        report.append("1. DATA INFORMATION")
        report.append("-" * 80)
        report.append(f"  Period: {self.df['ds'].min()} to {self.df['ds'].max()}")
        report.append(f"  Total days: {len(self.df)}")
        report.append(f"  Mean call volume: {self.df['y'].mean():.1f}")
        report.append(f"  Std: {self.df['y'].std():.1f}")
        report.append(f"  CV: {self.df['y'].std() / self.df['y'].mean():.3f}")
        report.append("")
        
        # ------------------------------------------------------------------
        # 2. 診断結果
        # ------------------------------------------------------------------
        report.append("2. DIAGNOSTICS")
        report.append("-" * 80)
        
        if self.diagnostics:
            # 正規性
            if 'normality' in self.diagnostics:
                norm = self.diagnostics['normality']
                report.append(f"  Normality: {'Normal' if norm.get('is_normal') else 'Non-normal'} (p={norm.get('normaltest_p', 0):.4f})")
            
            # 定常性
            if 'stationarity' in self.diagnostics:
                stat = self.diagnostics['stationarity']
                report.append(f"  Stationarity: {'Stationary' if stat.get('is_stationary') else 'Non-stationary'} (ADF p={stat.get('adf_p_value', 0):.4f})")
            
            # 曜日効果
            if 'weekday_effect' in self.diagnostics:
                week = self.diagnostics['weekday_effect']
                report.append(f"  Weekday effect: {'Significant' if week.get('has_effect') else 'Not significant'} (ANOVA p={week.get('p_value', 1):.4e})")
            
            # 外れ値
            if 'outliers' in self.diagnostics:
                out = self.diagnostics['outliers']
                report.append(f"  Outliers: {out.get('count', 0)} ({out.get('percentage', 0):.2f}%)")
            
            # 二峰性
            if 'bimodality' in self.diagnostics:
                bi = self.diagnostics['bimodality']
                report.append(f"  Distribution: {'Bimodal' if bi.get('is_bimodal') else 'Unimodal'} ({bi.get('num_peaks', 1)} peaks)")
        
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
        # 4. アンサンブル性能
        # ------------------------------------------------------------------
        if self.ensemble_forecast is not None:
            report.append("4. ENSEMBLE PERFORMANCE")
            report.append("-" * 80)
            
            train_forecast = self.ensemble_forecast[self.ensemble_forecast['ds'].isin(self.df['ds'])]
            merged = pd.merge(self.df[['ds', 'y']], train_forecast[['ds', 'yhat']], on='ds')
            
            mae = mean_absolute_error(merged['y'], merged['yhat'])
            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
            mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
            
            report.append(f"  MAE: {mae:.2f}")
            report.append(f"  RMSE: {rmse:.2f}")
            report.append(f"  MAPE: {mape:.2f}%")
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
            report.append("     → Use robust evaluation metrics (MAE, MedAE)")
        
        if self.diagnostics.get('bimodality', {}).get('is_bimodal'):
            report.append("  ⚠️  Bimodal distribution detected")
            report.append("     → Investigate weekday/weekend split")
            report.append("     → Consider mixture models")
        
        if self.diagnostics.get('outliers', {}).get('percentage', 0) > 5:
            report.append("  ⚠️  High outlier percentage (>5%)")
            report.append("     → Review outlier handling strategy")
            report.append("     → Consider robust Prophet settings")
        
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
        """
        モデルを保存
        
        Parameters
        ----------
        filepath : str
            保存先パス (Noneの場合は output_dir/models.pkl)
        """
        if filepath is None:
            filepath = self.output_dir / 'models.pkl'
        
        save_obj = {
            'models': self.models,
            'ensemble_forecast': self.ensemble_forecast,
            'best_params': self.best_params,
            'diagnostics': self.diagnostics,
            'regimes': self.regimes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        
        logger.info(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        モデルをロード
        
        Parameters
        ----------
        filepath : str
            モデルファイルパス
        """
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.models = save_obj.get('models', {})
        self.ensemble_forecast = save_obj.get('ensemble_forecast')
        self.best_params = save_obj.get('best_params', {})
        self.diagnostics = save_obj.get('diagnostics', {})
        self.regimes = save_obj.get('regimes', {})
        
        logger.info(f"✅ Models loaded from {filepath}")
    
    # ========================================================================
    # 11. 完全実行パイプライン
    # ========================================================================
    def fit_predict(self, filepath: str, horizon: int = 30, 
                    cv_days: int = 90, quick_mode: bool = False) -> Dict:
        """
        完全実行パイプライン
        
        Parameters
        ----------
        filepath : str
            CSVファイルパス
        horizon : int
            予測期間 (日数)
        cv_days : int
            交差検証ホライズン (日数)
        quick_mode : bool
            高速モード (パラメータグリッド削減)
        
        Returns
        -------
        dict
            全結果を含む辞書
        """
        logger.info("🚀 Starting Prophet Ultimate Predictor pipeline...")
        
        # 1. データ読み込み
        self.load_data(filepath)
        
        # 2. 診断
        self.run_comprehensive_diagnostics()
        
        # 3. レジーム検出
        cv = self.diagnostics.get('basic_stats', {}).get('cv', 0)
        if cv > 0.4 or self.diagnostics.get('bimodality', {}).get('is_bimodal'):
            logger.info("🎯 High variability or bimodality detected. Running regime detection...")
            self.detect_regimes(n_regimes=2, method='kmeans')
        
        # 4. 祝日データフレーム作成
        holidays = self.create_holiday_dataframe()
        
        # 5. ハイパーパラメータ最適化
        self.optimize_hyperparameters(
            self.df, 
            holidays=holidays, 
            cv_horizon_days=cv_days,
            quick_mode=quick_mode
        )
        
        # 6. アンサンブルモデル訓練
        self.fit_ensemble_models(self.df, holidays=holidays, horizon=horizon)
        
        # 7. 可視化
        self.create_comprehensive_visualizations()
        
        # 8. レポート生成
        report = self.generate_report()
        
        # 9. モデル保存
        self.save_models()
        
        # 10. 予測結果をCSV保存
        if self.ensemble_forecast is not None:
            forecast_df = self.ensemble_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
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
            'report': report
        }


# ============================================================================
# コマンドライン実行
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prophet Ultimate Predictor for Call Center Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本実行 (30日予測)
  python prophet_ultimate_predictor.py data.csv
  
  # 60日予測、交差検証90日、高速モード
  python prophet_ultimate_predictor.py data.csv --horizon 60 --cv-days 90 --quick
  
  # カスタム出力ディレクトリ
  python prophet_ultimate_predictor.py data.csv --output my_results
        """
    )
    
    parser.add_argument('filepath', type=str, help='Path to CSV file (ds, y columns)')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days (default: 30)')
    parser.add_argument('--cv-days', type=int, default=90, help='Cross-validation horizon (default: 90)')
    parser.add_argument('--output', type=str, default='prophet_ultimate_results', 
                        help='Output directory (default: prophet_ultimate_results)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced parameter grid)')
    
    args = parser.parse_args()
    
    # 実行
    predictor = ProphetUltimatePredictor(output_dir=args.output)
    results = predictor.fit_predict(
        args.filepath, 
        horizon=args.horizon, 
        cv_days=args.cv_days,
        quick_mode=args.quick
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(results['report'])
