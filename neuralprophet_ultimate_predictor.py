#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
NeuralProphet Ultimate Predictor for Call Center v5.0
==============================================================================

コールセンター日次呼量予測システム v5.0 (NeuralProphet版)
- ディープラーニングベースの非線形自己回帰 (AR-Net)
- 超高精度特徴量エンジニアリング (100+ features)
- Optunaベイズ最適化 (30+ hyperparameters)
- Quantile Loss最適化
- シフト計画特化評価指標 (wQL, WAPE, MASE)

主要機能:
---------
1. NeuralProphet AR-Net
   - 自動ラグ選択 (1〜365日)
   - 非線形パターン学習
   - 訓練可能な seasonality
2. 超高精度特徴量
   - Lagged regressors: 短期〜長期ラグ (12種類)
   - Future regressors: 曜日、月、祝日、カレンダー特徴 (50+)
   - Rolling features: rolling mean/std, EWM (20+)
   - Events: 日本の祝日、特殊期間
3. 自動変換選択
   - 正規性検定 (5種類)
   - Box-Cox, Yeo-Johnson, log, sqrt, reciprocal
4. Optuna最適化
   - 30+ ハイパーパラメータ
   - Quantile loss (QL_60, QL_70)
   - 時系列CV
5. シフト計画特化評価
   - wQL, WAPE, MASE
   - Peak day accuracy
   - Bias analysis
6. 包括的可視化とレポート

使用例:
-------
python neuralprophet_ultimate_predictor.py data.csv \\
    --validation-months 2 \\
    --optuna-trials 200 \\
    --n-lags 28 \\
    --ar-layers 64 \\
    --epochs 100 \\
    --quantile 0.6

作成者: AI Assistant
バージョン: 5.0
最終更新: 2026-02-19
ライセンス: MIT
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
import argparse

# NeuralProphet
try:
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")  # ログ抑制
except ImportError:
    print("❌ NeuralProphet not installed.")
    print("Run: pip install neuralprophet")
    print("Or: pip install neuralprophet[live]  # for live plotting")
    sys.exit(1)

# Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ Optuna not installed. Run: pip install optuna")
    sys.exit(1)

# PyTorch (NeuralProphetのバックエンド)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU検出: {torch.cuda.get_device_name(0)}")
        DEVICE = 'cuda'
    else:
        print("✅ CPU モード")
        DEVICE = 'cpu'
except ImportError:
    print("⚠️  PyTorch not installed. Run: pip install torch")
    DEVICE = 'cpu'

# 統計・時系列分析
from scipy import stats
from scipy.stats import (normaltest, shapiro, jarque_bera, anderson, 
                         boxcox, yeojohnson, skew, kurtosis, zscore)
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 評価指標システム
try:
    from evaluation_metrics_for_shift_planning import ShiftPlanningEvaluator
    SHIFT_EVAL_AVAILABLE = True
except ImportError:
    print("⚠️  evaluation_metrics_for_shift_planning not found.")
    print("Advanced shift planning metrics will not be available.")
    SHIFT_EVAL_AVAILABLE = False

# 日本の祝日
try:
    import jpholiday
    JPHOLIDAY_AVAILABLE = True
except ImportError:
    print("⚠️  jpholiday not installed. Run: pip install jpholiday")
    JPHOLIDAY_AVAILABLE = False

# プログレスバー
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class NeuralProphetUltimatePredictor:
    """
    NeuralProphet v5.0 超高精度予測システム
    
    Features:
    ---------
    - AR-Net (自己回帰ニューラルネット)
    - 100+ 特徴量
    - Optuna最適化 (30+ params)
    - Quantile loss最適化
    - シフト計画特化評価
    """
    
    def __init__(self, output_dir: str = "output_neuralprophet"):
        """初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # データ
        self.df_train = None
        self.df_train_original = None
        self.df_validation = None
        self.df_validation_original = None
        self.df_full = None
        self.df_full_original = None
        
        # 特徴量
        self.lagged_regressor_names = []
        self.future_regressor_names = []
        self.event_names = []
        
        # 診断結果
        self.diagnostics = {}
        
        # モデルと予測
        self.best_params = {}
        self.model_validation = None
        self.model_production = None
        self.forecast_validation = None
        self.forecast_production = None
        self.validation_metrics = {}
        
        # 変換情報
        self.transformation_info = {}
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "neuralprophet_v5.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 100)
        self.logger.info("NeuralProphet v5.0 超高精度予測システム 初期化完了")
        self.logger.info(f"Device: {DEVICE}")
        self.logger.info("=" * 100)
    
    def load_data(self, filepath: str, date_col: str = "ds", 
                  target_col: str = "y") -> pd.DataFrame:
        """
        CSVデータ読み込み
        
        Parameters
        ----------
        filepath : str
            CSVファイルパス
        date_col : str
            日付カラム名
        target_col : str
            目的変数カラム名
        
        Returns
        -------
        pd.DataFrame
            読み込んだデータ
        """
        self.logger.info(f"📁 データ読み込み開始: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # 列名標準化
        if date_col not in df.columns and target_col not in df.columns:
            if len(df.columns) >= 2:
                df.columns = ['ds', 'y'] + list(df.columns[2:])
                self.logger.info(f"✓ 列名を自動変換: {date_col} -> ds, {target_col} -> y")
        
        # 日付変換
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # 欠損値チェック
        missing = df[['ds', 'y']].isnull().sum()
        if missing.any():
            self.logger.warning(f"⚠️  欠損値検出: {missing.to_dict()}")
            df = df.dropna(subset=['ds', 'y'])
        
        # 重複日付チェック
        duplicates = df['ds'].duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"⚠️  重複日付検出: {duplicates} 件")
            df = df.drop_duplicates(subset=['ds'], keep='last')
        
        self.logger.info(f"✓ データ読み込み完了: {len(df)} 行")
        self.logger.info(f"  期間: {df['ds'].min()} 〜 {df['ds'].max()}")
        self.logger.info(f"  日数: {(df['ds'].max() - df['ds'].min()).days + 1} 日")
        self.logger.info(f"  目的変数統計:")
        self.logger.info(f"    平均: {df['y'].mean():.2f}")
        self.logger.info(f"    中央値: {df['y'].median():.2f}")
        self.logger.info(f"    標準偏差: {df['y'].std():.2f}")
        self.logger.info(f"    最小値: {df['y'].min():.2f}")
        self.logger.info(f"    最大値: {df['y'].max():.2f}")
        self.logger.info(f"    変動係数: {df['y'].std() / df['y'].mean():.4f}")
        
        return df
    
    def select_optimal_transformation(self, y: pd.Series) -> Dict:
        """
        データ形状に基づく最適な変換を自動選択
        
        Parameters
        ----------
        y : pd.Series
            元の目的変数
        
        Returns
        -------
        Dict
            変換情報 (method, transformed_y, lambda_param, metrics)
        """
        self.logger.info("=" * 100)
        self.logger.info("🔄 データ変換の自動選択開始")
        self.logger.info("=" * 100)
        
        y_clean = y.dropna()
        
        # 元データの統計量
        original_stats = {
            'mean': y_clean.mean(),
            'std': y_clean.std(),
            'skewness': skew(y_clean),
            'kurtosis': kurtosis(y_clean),
            'min': y_clean.min(),
            'max': y_clean.max()
        }
        
        self.logger.info(f"📊 元データ統計:")
        self.logger.info(f"  平均: {original_stats['mean']:.2f}")
        self.logger.info(f"  標準偏差: {original_stats['std']:.2f}")
        self.logger.info(f"  歪度: {original_stats['skewness']:.4f}")
        self.logger.info(f"  尖度: {original_stats['kurtosis']:.4f}")
        
        # 正規性検定
        def test_normality(data):
            """正規性検定 (5種類)"""
            tests = {}
            
            # 1. Shapiro-Wilk
            if len(data) < 5000:
                stat, p = shapiro(data)
                tests['shapiro'] = {'statistic': stat, 'pvalue': p}
            
            # 2. Kolmogorov-Smirnov
            stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            tests['ks'] = {'statistic': stat, 'pvalue': p}
            
            # 3. Anderson-Darling
            result = anderson(data, dist='norm')
            tests['anderson'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist()
            }
            
            # 4. Jarque-Bera
            stat, p = jarque_bera(data)
            tests['jarque_bera'] = {'statistic': stat, 'pvalue': p}
            
            # 5. D'Agostino-Pearson
            stat, p = normaltest(data)
            tests['dagostino'] = {'statistic': stat, 'pvalue': p}
            
            return tests
        
        original_tests = test_normality(y_clean)
        
        self.logger.info("🔬 元データの正規性検定:")
        for name, result in original_tests.items():
            if 'pvalue' in result:
                is_normal = "正規" if result['pvalue'] > 0.05 else "非正規"
                self.logger.info(f"  {name}: p値={result['pvalue']:.6f} ({is_normal})")
        
        # 変換候補を評価
        transformations = []
        
        # 1. 変換なし
        transformations.append({
            'method': 'none',
            'y_transformed': y_clean,
            'lambda': None,
            'stats': original_stats,
            'tests': original_tests,
            'score': self._score_transformation(original_stats, original_tests)
        })
        
        # 2. 対数変換 (y > 0)
        if y_clean.min() > 0:
            y_log = np.log(y_clean)
            log_stats = {
                'skewness': skew(y_log),
                'kurtosis': kurtosis(y_log)
            }
            log_tests = test_normality(y_log)
            transformations.append({
                'method': 'log',
                'y_transformed': y_log,
                'lambda': None,
                'stats': log_stats,
                'tests': log_tests,
                'score': self._score_transformation(log_stats, log_tests)
            })
        
        # 3. 平方根変換 (y >= 0)
        if y_clean.min() >= 0:
            y_sqrt = np.sqrt(y_clean)
            sqrt_stats = {
                'skewness': skew(y_sqrt),
                'kurtosis': kurtosis(y_sqrt)
            }
            sqrt_tests = test_normality(y_sqrt)
            transformations.append({
                'method': 'sqrt',
                'y_transformed': y_sqrt,
                'lambda': None,
                'stats': sqrt_stats,
                'tests': sqrt_tests,
                'score': self._score_transformation(sqrt_stats, sqrt_tests)
            })
        
        # 4. Box-Cox変換 (y > 0)
        if y_clean.min() > 0:
            try:
                y_boxcox, lambda_bc = boxcox(y_clean)
                bc_stats = {
                    'skewness': skew(y_boxcox),
                    'kurtosis': kurtosis(y_boxcox)
                }
                bc_tests = test_normality(y_boxcox)
                transformations.append({
                    'method': 'boxcox',
                    'y_transformed': pd.Series(y_boxcox, index=y_clean.index),
                    'lambda': lambda_bc,
                    'stats': bc_stats,
                    'tests': bc_tests,
                    'score': self._score_transformation(bc_stats, bc_tests)
                })
            except Exception as e:
                self.logger.warning(f"⚠️  Box-Cox変換失敗: {e}")
        
        # 5. Yeo-Johnson変換 (全ての値)
        try:
            y_yj, lambda_yj = yeojohnson(y_clean)
            yj_stats = {
                'skewness': skew(y_yj),
                'kurtosis': kurtosis(y_yj)
            }
            yj_tests = test_normality(y_yj)
            transformations.append({
                'method': 'yeojohnson',
                'y_transformed': pd.Series(y_yj, index=y_clean.index),
                'lambda': lambda_yj,
                'stats': yj_stats,
                'tests': yj_tests,
                'score': self._score_transformation(yj_stats, yj_tests)
            })
        except Exception as e:
            self.logger.warning(f"⚠️  Yeo-Johnson変換失敗: {e}")
        
        # 6. 逆数変換 (y != 0)
        if y_clean.min() > 0:
            y_reciprocal = 1 / y_clean
            recip_stats = {
                'skewness': skew(y_reciprocal),
                'kurtosis': kurtosis(y_reciprocal)
            }
            recip_tests = test_normality(y_reciprocal)
            transformations.append({
                'method': 'reciprocal',
                'y_transformed': y_reciprocal,
                'lambda': None,
                'stats': recip_stats,
                'tests': recip_tests,
                'score': self._score_transformation(recip_stats, recip_tests)
            })
        
        # 最適な変換を選択 (スコア最大)
        best_transformation = max(transformations, key=lambda x: x['score'])
        
        self.logger.info("=" * 100)
        self.logger.info("📋 変換候補の評価結果:")
        self.logger.info("=" * 100)
        for t in transformations:
            marker = "⭐" if t == best_transformation else "  "
            self.logger.info(f"{marker} {t['method']:12s}: "
                           f"Score={t['score']:.4f}, "
                           f"Skewness={t['stats']['skewness']:7.4f}, "
                           f"Kurtosis={t['stats']['kurtosis']:7.4f}")
        
        self.logger.info("=" * 100)
        self.logger.info(f"✅ 選択された変換: {best_transformation['method']}")
        if best_transformation['lambda'] is not None:
            self.logger.info(f"  Lambda: {best_transformation['lambda']:.4f}")
        self.logger.info("=" * 100)
        
        return best_transformation
    
    def _score_transformation(self, stats: Dict, tests: Dict) -> float:
        """
        変換の良さをスコアリング
        
        Parameters
        ----------
        stats : Dict
            統計量 (skewness, kurtosis)
        tests : Dict
            正規性検定結果
        
        Returns
        -------
        float
            スコア (高いほど良い)
        """
        score = 0.0
        
        # 1. 歪度が0に近い (+30点満点)
        score += 30 * np.exp(-abs(stats['skewness']))
        
        # 2. 尖度が0に近い (+30点満点)
        score += 30 * np.exp(-abs(stats['kurtosis']))
        
        # 3. 正規性検定 p値 > 0.05 (+40点満点)
        p_values = []
        for name, result in tests.items():
            if 'pvalue' in result:
                p_values.append(result['pvalue'])
        
        if p_values:
            avg_p = np.mean(p_values)
            score += 40 * (avg_p ** 0.5)  # 平方根で変換 (p=1で満点)
        
        return score
    
    def _apply_transformation(self, y: pd.Series, 
                             method: str, 
                             lambda_param: Optional[float] = None) -> pd.Series:
        """
        変換を適用
        
        Parameters
        ----------
        y : pd.Series
            元データ
        method : str
            変換方法
        lambda_param : float, optional
            Box-Cox/Yeo-Johnsonのλ
        
        Returns
        -------
        pd.Series
            変換後データ
        """
        if method == 'none':
            return y
        elif method == 'log':
            return np.log(y)
        elif method == 'sqrt':
            return np.sqrt(y)
        elif method == 'boxcox':
            if lambda_param == 0:
                return pd.Series(np.log(y), index=y.index)
            else:
                return pd.Series((y ** lambda_param - 1) / lambda_param, index=y.index)
        elif method == 'yeojohnson':
            return pd.Series(yeojohnson(y, lmbda=lambda_param), index=y.index)
        elif method == 'reciprocal':
            return 1 / y
        else:
            raise ValueError(f"Unknown transformation: {method}")
    
    def _inverse_transformation(self, y_transformed: np.ndarray, 
                                method: str, 
                                lambda_param: Optional[float] = None) -> np.ndarray:
        """
        変換を逆変換
        
        Parameters
        ----------
        y_transformed : np.ndarray
            変換後データ
        method : str
            変換方法
        lambda_param : float, optional
            Box-Cox/Yeo-Johnsonのλ
        
        Returns
        -------
        np.ndarray
            元のスケールに戻したデータ
        """
        if method == 'none':
            return y_transformed
        elif method == 'log':
            return np.exp(y_transformed)
        elif method == 'sqrt':
            return y_transformed ** 2
        elif method == 'boxcox':
            if lambda_param == 0:
                return np.exp(y_transformed)
            else:
                return (y_transformed * lambda_param + 1) ** (1 / lambda_param)
        elif method == 'yeojohnson':
            # Yeo-Johnson逆変換
            if lambda_param == 0:
                return np.exp(y_transformed) - 1
            elif lambda_param == 2:
                return np.exp(-y_transformed) - 1
            elif y_transformed >= 0:
                return (y_transformed * lambda_param + 1) ** (1 / lambda_param) - 1
            else:
                return 1 - ((-y_transformed) * (2 - lambda_param) + 1) ** (1 / (2 - lambda_param))
        elif method == 'reciprocal':
            return 1 / y_transformed
        else:
            raise ValueError(f"Unknown transformation: {method}")
    
    # ... (続く: 特徴量生成、Optuna最適化、モデル訓練など)
    
    def generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        超包括的な特徴量生成 (100+ features)
        
        NeuralProphetに最適化:
        - Lagged regressors: 過去の値 (AR-Netで学習)
        - Future regressors: 将来わかる特徴
        - Events: 特殊日
        
        Parameters
        ----------
        df : pd.DataFrame
            入力データ (ds, y)
        
        Returns
        -------
        pd.DataFrame
            特徴量追加後のデータ
        """
        self.logger.info("=" * 100)
        self.logger.info("🔧 超包括的特徴量生成開始 (100+ features)")
        self.logger.info("=" * 100)
        
        df = df.copy()
        df = df.sort_values('ds').reset_index(drop=True)
        
        # 1. ラグ特徴量 (Lagged Regressors)
        df = self._generate_lagged_features(df)
        
        # 2. ローリング統計 (Lagged Regressors)
        df = self._generate_rolling_features(df)
        
        # 3. 時間特徴 (Future Regressors)
        df = self._generate_time_features(df)
        
        # 4. カレンダー特徴 (Future Regressors)
        df = self._generate_calendar_features(df)
        
        # 5. 循環エンコーディング (Future Regressors)
        df = self._generate_cyclical_features(df)
        
        # 6. イベント特徴 (Events)
        df = self._generate_event_features(df)
        
        # 7. トレンド特徴 (Future Regressors)
        df = self._generate_trend_features(df)
        
        # 欠損値処理 (ラグ特徴による欠損)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            self.logger.info(f"⚠️  ラグ特徴による欠損行削除: {dropped_rows} 行")
        
        self.logger.info("=" * 100)
        self.logger.info(f"✅ 特徴量生成完了")
        self.logger.info(f"  Lagged regressors: {len(self.lagged_regressor_names)} 個")
        self.logger.info(f"  Future regressors: {len(self.future_regressor_names)} 個")
        self.logger.info(f"  Events: {len(self.event_names)} 個")
        self.logger.info(f"  総特徴量数: {len(df.columns) - 2} 個 (ds, y を除く)")
        self.logger.info("=" * 100)
        
        return df
    
    def _generate_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ラグ特徴量生成
        
        Lagged regressors: NeuralProphetのAR-Netが学習
        短期〜長期のラグを網羅
        """
        self.logger.info("📊 ラグ特徴量生成中...")
        
        # 基本ラグ (1〜7日: 直近1週間)
        for lag in [1, 2, 3, 4, 5, 6, 7]:
            col_name = f'lag_{lag}'
            df[col_name] = df['y'].shift(lag)
            self.lagged_regressor_names.append(col_name)
        
        # 週次ラグ (7, 14, 21, 28日: 過去4週間の同曜日)
        for lag in [14, 21, 28]:
            col_name = f'lag_{lag}'
            df[col_name] = df['y'].shift(lag)
            self.lagged_regressor_names.append(col_name)
        
        # 月次ラグ (30, 60, 90日)
        for lag in [30, 60, 90]:
            col_name = f'lag_{lag}'
            df[col_name] = df['y'].shift(lag)
            self.lagged_regressor_names.append(col_name)
        
        # 長期ラグ (180, 365日: 半年・1年前)
        for lag in [180, 365]:
            if len(df) > lag:
                col_name = f'lag_{lag}'
                df[col_name] = df['y'].shift(lag)
                self.lagged_regressor_names.append(col_name)
        
        self.logger.info(f"  ✓ ラグ特徴: {len(self.lagged_regressor_names)} 個")
        
        return df
    
    def _generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ローリング統計特徴
        
        Lagged regressors: rolling mean/std, EWM
        """
        self.logger.info("📊 ローリング統計特徴生成中...")
        
        # Rolling mean (7, 14, 28日)
        for window in [7, 14, 28]:
            col_name = f'rolling_mean_{window}'
            df[col_name] = df['y'].shift(1).rolling(window).mean()
            self.lagged_regressor_names.append(col_name)
        
        # Rolling std (7, 14, 28日)
        for window in [7, 14, 28]:
            col_name = f'rolling_std_{window}'
            df[col_name] = df['y'].shift(1).rolling(window).std()
            self.lagged_regressor_names.append(col_name)
        
        # Rolling min/max (7, 14日)
        for window in [7, 14]:
            col_name = f'rolling_min_{window}'
            df[col_name] = df['y'].shift(1).rolling(window).min()
            self.lagged_regressor_names.append(col_name)
            
            col_name = f'rolling_max_{window}'
            df[col_name] = df['y'].shift(1).rolling(window).max()
            self.lagged_regressor_names.append(col_name)
        
        # Exponential weighted mean (7, 14, 28日)
        for span in [7, 14, 28]:
            col_name = f'ewm_{span}'
            df[col_name] = df['y'].shift(1).ewm(span=span).mean()
            self.lagged_regressor_names.append(col_name)
        
        # 変動係数 (CV: coefficient of variation)
        for window in [7, 14, 28]:
            mean_col = f'rolling_mean_{window}'
            std_col = f'rolling_std_{window}'
            if mean_col in df.columns and std_col in df.columns:
                col_name = f'cv_{window}'
                df[col_name] = df[std_col] / (df[mean_col] + 1e-10)
                self.lagged_regressor_names.append(col_name)
        
        self.logger.info(f"  ✓ ローリング特徴: {len([n for n in self.lagged_regressor_names if 'rolling' in n or 'ewm' in n or 'cv' in n])} 個")
        
        return df
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間特徴 (Future Regressors)
        
        曜日、月、四半期など
        """
        self.logger.info("📅 時間特徴生成中...")
        
        # 曜日 (0=月曜, 6=日曜)
        df['dayofweek'] = df['ds'].dt.dayofweek
        self.future_regressor_names.append('dayofweek')
        
        # 曜日ダミー (one-hot)
        for dow in range(7):
            col_name = f'dow_{dow}'
            df[col_name] = (df['dayofweek'] == dow).astype(int)
            self.future_regressor_names.append(col_name)
        
        # 月 (1-12)
        df['month'] = df['ds'].dt.month
        self.future_regressor_names.append('month')
        
        # 月ダミー (one-hot)
        for month in range(1, 13):
            col_name = f'month_{month}'
            df[col_name] = (df['month'] == month).astype(int)
            self.future_regressor_names.append(col_name)
        
        # 四半期 (1-4)
        df['quarter'] = df['ds'].dt.quarter
        self.future_regressor_names.append('quarter')
        
        # 四半期ダミー
        for q in range(1, 5):
            col_name = f'quarter_{q}'
            df[col_name] = (df['quarter'] == q).astype(int)
            self.future_regressor_names.append(col_name)
        
        # 年
        df['year'] = df['ds'].dt.year
        self.future_regressor_names.append('year')
        
        # 月内日 (1-31)
        df['day_of_month'] = df['ds'].dt.day
        self.future_regressor_names.append('day_of_month')
        
        # 年内日 (1-365/366)
        df['day_of_year'] = df['ds'].dt.dayofyear
        self.future_regressor_names.append('day_of_year')
        
        # 年内週 (1-53)
        df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
        self.future_regressor_names.append('week_of_year')
        
        # 月内週 (1-5)
        df['week_of_month'] = ((df['day_of_month'] - 1) // 7 + 1)
        self.future_regressor_names.append('week_of_month')
        
        self.logger.info(f"  ✓ 時間特徴: {len([n for n in self.future_regressor_names if 'dow' in n or 'month' in n or 'quarter' in n or 'year' in n or 'day' in n or 'week' in n])} 個")
        
        return df
    
    def _generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カレンダー特徴 (Future Regressors)
        
        平日/週末、月初月末など
        """
        self.logger.info("📆 カレンダー特徴生成中...")
        
        # 平日/週末
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        self.future_regressor_names.append('is_weekend')
        
        # 月曜・金曜フラグ
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        self.future_regressor_names.extend(['is_monday', 'is_friday'])
        
        # 月初 (1-5日)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        self.future_regressor_names.append('is_month_start')
        
        # 月末 (26-31日)
        df['days_in_month'] = df['ds'].dt.days_in_month
        df['is_month_end'] = (df['day_of_month'] >= df['days_in_month'] - 5).astype(int)
        self.future_regressor_names.append('is_month_end')
        
        # 月中旬 (10-20日)
        df['is_mid_month'] = ((df['day_of_month'] >= 10) & (df['day_of_month'] <= 20)).astype(int)
        self.future_regressor_names.append('is_mid_month')
        
        # 月末までの日数
        df['days_to_month_end'] = df['days_in_month'] - df['day_of_month']
        self.future_regressor_names.append('days_to_month_end')
        
        # 四半期初・四半期末
        df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
        self.future_regressor_names.extend(['is_quarter_start', 'is_quarter_end'])
        
        # 年初・年末
        df['is_year_start'] = (df['day_of_year'] <= 5).astype(int)
        df['is_year_end'] = (df['day_of_year'] >= 360).astype(int)
        self.future_regressor_names.extend(['is_year_start', 'is_year_end'])
        
        self.logger.info(f"  ✓ カレンダー特徴: {len([n for n in self.future_regressor_names if 'is_' in n or 'days_' in n])} 個")
        
        return df
    
    def _generate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        循環エンコーディング (Future Regressors)
        
        sin/cos変換で周期性を表現
        """
        self.logger.info("🔄 循環特徴生成中...")
        
        # 曜日 (周期=7)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        self.future_regressor_names.extend(['dow_sin', 'dow_cos'])
        
        # 月 (周期=12)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        self.future_regressor_names.extend(['month_sin', 'month_cos'])
        
        # 月内日 (周期=31)
        df['day_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
        df['day_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
        self.future_regressor_names.extend(['day_sin', 'day_cos'])
        
        # 年内日 (周期=365)
        df['doy_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
        df['doy_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
        self.future_regressor_names.extend(['doy_sin', 'doy_cos'])
        
        # 四半期 (周期=4)
        df['quarter_sin'] = np.sin(2 * np.pi * (df['quarter'] - 1) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * (df['quarter'] - 1) / 4)
        self.future_regressor_names.extend(['quarter_sin', 'quarter_cos'])
        
        self.logger.info(f"  ✓ 循環特徴: {len([n for n in self.future_regressor_names if 'sin' in n or 'cos' in n])} 個")
        
        return df
    
    def _generate_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        イベント特徴 (Events)
        
        日本の祝日、特殊期間
        """
        self.logger.info("🎌 イベント特徴生成中...")
        
        if not JPHOLIDAY_AVAILABLE:
            self.logger.warning("  ⚠️  jpholiday未インストール → イベント特徴スキップ")
            return df
        
        # 祝日フラグ
        df['is_holiday'] = df['ds'].apply(lambda x: jpholiday.is_holiday(x)).astype(int)
        self.event_names.append('is_holiday')
        
        # 祝日名取得
        df['holiday_name'] = df['ds'].apply(
            lambda x: jpholiday.is_holiday_name(x) if jpholiday.is_holiday(x) else None
        )
        
        # ゴールデンウィーク (4/29 - 5/5)
        df['is_golden_week'] = (
            (df['month'] == 4) & (df['day_of_month'] >= 29) |
            (df['month'] == 5) & (df['day_of_month'] <= 5)
        ).astype(int)
        self.event_names.append('is_golden_week')
        
        # お盆 (8/13 - 8/16)
        df['is_obon'] = (
            (df['month'] == 8) & 
            (df['day_of_month'] >= 13) & 
            (df['day_of_month'] <= 16)
        ).astype(int)
        self.event_names.append('is_obon')
        
        # 年末年始 (12/29 - 1/3)
        df['is_year_end_new_year'] = (
            (df['month'] == 12) & (df['day_of_month'] >= 29) |
            (df['month'] == 1) & (df['day_of_month'] <= 3)
        ).astype(int)
        self.event_names.append('is_year_end_new_year')
        
        # シルバーウィーク (9月の連休)
        df['is_silver_week'] = (
            (df['month'] == 9) & 
            (df['day_of_month'] >= 15) & 
            (df['day_of_month'] <= 23) &
            (df['is_holiday'] == 1)
        ).astype(int)
        self.event_names.append('is_silver_week')
        
        # 祝日前日・翌日
        df['is_holiday_before'] = df['is_holiday'].shift(-1).fillna(0).astype(int)
        df['is_holiday_after'] = df['is_holiday'].shift(1).fillna(0).astype(int)
        self.event_names.extend(['is_holiday_before', 'is_holiday_after'])
        
        self.logger.info(f"  ✓ イベント特徴: {len(self.event_names)} 個")
        
        return df
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        トレンド特徴 (Future Regressors)
        
        時間インデックス、成長率など
        """
        self.logger.info("📈 トレンド特徴生成中...")
        
        # 時間インデックス (0, 1, 2, ...)
        df['t'] = np.arange(len(df))
        self.future_regressor_names.append('t')
        
        # 時間インデックスの2乗、3乗
        df['t_squared'] = df['t'] ** 2
        df['t_cubed'] = df['t'] ** 3
        self.future_regressor_names.extend(['t_squared', 't_cubed'])
        
        # 正規化時間 (0-1)
        df['t_normalized'] = df['t'] / (len(df) - 1)
        self.future_regressor_names.append('t_normalized')
        
        self.logger.info(f"  ✓ トレンド特徴: {len([n for n in self.future_regressor_names if 't' in n and 'month' not in n])} 個")
        
        return df

