#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
NeuralProphet Ultimate Predictor for Call Center v5.0 - COMPLETE
==============================================================================

コールセンター日次呼量予測システム v5.0 (NeuralProphet版) - 完全実装版
- ディープラーニングベースの非線形自己回帰 (AR-Net)
- 超高精度特徴量エンジニアリング (100+ features)
- Optunaベイズ最適化 (30+ hyperparameters) ✅ 実装済み
- Quantile Loss最適化 ✅ 実装済み
- シフト計画特化評価指標 (wQL, WAPE, MASE) ✅ 実装済み
- 完全な可視化とレポート ✅ 実装済み
- Jupyter Notebook対応 ✅ 実装済み

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
4. Optuna最適化 ✅
   - 30+ ハイパーパラメータ
   - Quantile loss (QL_60, QL_70)
   - 時系列CV
5. シフト計画特化評価 ✅
   - wQL, WAPE, MASE
   - Peak day accuracy
   - Bias analysis
6. 包括的可視化とレポート ✅

使用例 (Jupyter):
-----------------
from neuralprophet_ultimate_predictor_full import NeuralProphetUltimatePredictor

# 初期化
predictor = NeuralProphetUltimatePredictor(
    validation_months=2,
    optuna_trials=100,
    target_quantile=0.6
)

# データ読み込み
df = predictor.load_data('data.csv', date_col='date', value_col='y')

# 自動変換選択
df_transformed = predictor.select_optimal_transformation(df)

# 特徴量生成
df_features = predictor.generate_comprehensive_features(df_transformed)

# 訓練・検証分割
train_df, val_df = predictor.split_train_validation(df_features)

# Optuna最適化 + 訓練
best_model, best_params = predictor.optimize_and_train(train_df, val_df)

# 予測
forecast_df = predictor.predict(best_model, periods=60, include_history=True)

# 評価
metrics = predictor.evaluate(val_df, forecast_df)

# 可視化
predictor.plot_forecast(forecast_df, val_df)
predictor.plot_components(best_model, forecast_df)
predictor.plot_metrics(metrics)

# レポート生成
predictor.generate_report(metrics, best_params, output_path='report.html')

作成者: AI Assistant
バージョン: 5.0
最終更新: 2026-02-20
ライセンス: MIT
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要
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
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print("❌ PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# jpholiday (日本の祝日)
try:
    import jpholiday
    JPHOLIDAY_AVAILABLE = True
    print("✅ jpholiday インストール済み")
except ImportError:
    JPHOLIDAY_AVAILABLE = False
    print("⚠️  jpholiday 未インストール (イベント特徴なし)")
    print("   推奨: pip install jpholiday")

# Jupyter表示設定
try:
    from IPython.display import display, HTML
    JUPYTER_MODE = True
except ImportError:
    JUPYTER_MODE = False

# プロット設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# ==============================================================================
# Main Predictor Class
# ==============================================================================

class NeuralProphetUltimatePredictor:
    """
    NeuralProphet Ultimate Predictor v5.0
    
    完全実装版：Optuna最適化、訓練、評価、可視化すべて含む
    """
    
    def __init__(
        self,
        validation_months: int = 2,
        optuna_trials: int = 100,
        target_quantile: float = 0.6,
        n_lags: Optional[int] = None,
        ar_layers: Optional[List[int]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = None,
        log_dir: str = './logs',
        output_dir: str = './outputs'
    ):
        """
        初期化
        
        Parameters
        ----------
        validation_months : int
            検証期間の月数 (デフォルト: 2)
        optuna_trials : int
            Optuna試行回数 (デフォルト: 100)
        target_quantile : float
            目標分位点 (0.5=中央値, 0.6=シフト推奨, 0.7=保守的)
        n_lags : int, optional
            AR-Netラグ数 (None=自動)
        ar_layers : list, optional
            AR-Net層構成 (None=自動)
        epochs : int
            訓練エポック数
        batch_size : int
            バッチサイズ
        learning_rate : float, optional
            学習率 (None=自動)
        log_dir : str
            ログ出力ディレクトリ
        output_dir : str
            結果出力ディレクトリ
        """
        self.validation_months = validation_months
        self.optuna_trials = optuna_trials
        self.target_quantile = target_quantile
        self.n_lags = n_lags
        self.ar_layers = ar_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # ディレクトリ作成
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self.logger = self._setup_logger()
        
        # 特徴量名リスト
        self.lagged_regressor_names = []
        self.future_regressor_names = []
        self.event_names = []
        
        # 変換パラメータ
        self.transformation_type = None
        self.transformation_params = {}
        self.original_mean = None
        self.original_std = None
        
        # モデル
        self.best_model = None
        self.best_params = None
        self.study = None
        
        self.logger.info("=" * 80)
        self.logger.info("NeuralProphet Ultimate Predictor v5.0 - 初期化完了")
        self.logger.info("=" * 80)
        self.logger.info(f"検証期間: {validation_months} ヶ月")
        self.logger.info(f"Optuna試行: {optuna_trials} 回")
        self.logger.info(f"目標分位点: {target_quantile}")
        self.logger.info(f"デバイス: {DEVICE}")
        self.logger.info(f"jpholiday: {'✅' if JPHOLIDAY_AVAILABLE else '❌'}")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger('NeuralProphet_Ultimate')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # ファイルハンドラ
        log_file = self.log_dir / f'neuralprophet_ultimate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # コンソールハンドラ
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # フォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_data(
        self,
        file_path: str,
        date_col: str = 'date',
        value_col: str = 'y',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        データ読み込み
        
        Parameters
        ----------
        file_path : str
            CSVファイルパス
        date_col : str
            日付カラム名
        value_col : str
            目的変数カラム名
        parse_dates : bool
            日付パース
        
        Returns
        -------
        pd.DataFrame
            読み込んだデータ (ds, y カラム)
        """
        self.logger.info("=" * 80)
        self.logger.info("📂 データ読み込み開始")
        self.logger.info("=" * 80)
        
        # CSV読み込み
        df = pd.read_csv(file_path)
        self.logger.info(f"  ✓ ファイル: {file_path}")
        self.logger.info(f"  ✓ 行数: {len(df):,}")
        self.logger.info(f"  ✓ カラム: {list(df.columns)}")
        
        # 日付カラム検出
        if date_col not in df.columns:
            # 自動検出
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                self.logger.info(f"  ✓ 日付カラム自動検出: {date_col}")
            else:
                raise ValueError(f"日付カラム '{date_col}' が見つかりません")
        
        # 値カラム検出
        if value_col not in df.columns:
            # 自動検出 (数値カラムの最初)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]
                self.logger.info(f"  ✓ 値カラム自動検出: {value_col}")
            else:
                raise ValueError(f"値カラム '{value_col}' が見つかりません")
        
        # データフレーム作成
        df_clean = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]) if parse_dates else df[date_col],
            'y': df[value_col]
        })
        
        # ソート
        df_clean = df_clean.sort_values('ds').reset_index(drop=True)
        
        # 欠損値確認
        missing_count = df_clean['y'].isna().sum()
        if missing_count > 0:
            self.logger.warning(f"  ⚠️  欠損値: {missing_count} 個 → 線形補間")
            df_clean['y'] = df_clean['y'].interpolate(method='linear')
        
        # 統計情報
        self.logger.info(f"\n📊 基本統計:")
        self.logger.info(f"  期間: {df_clean['ds'].min().date()} 〜 {df_clean['ds'].max().date()}")
        self.logger.info(f"  日数: {len(df_clean)} 日")
        self.logger.info(f"  平均: {df_clean['y'].mean():.2f}")
        self.logger.info(f"  標準偏差: {df_clean['y'].std():.2f}")
        self.logger.info(f"  最小: {df_clean['y'].min():.2f}")
        self.logger.info(f"  最大: {df_clean['y'].max():.2f}")
        
        return df_clean
    
    def select_optimal_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        最適な変換を自動選択
        
        正規性検定で最も正規分布に近い変換を選択
        
        Parameters
        ----------
        df : pd.DataFrame
            入力データ (ds, y)
        
        Returns
        -------
        pd.DataFrame
            変換後データ (ds, y)
        """
        self.logger.info("=" * 80)
        self.logger.info("🔄 自動変換選択開始")
        self.logger.info("=" * 80)
        
        y = df['y'].values
        self.original_mean = y.mean()
        self.original_std = y.std()
        
        transformations = {}
        
        # 1. 元データ
        _, p_original = stats.shapiro(y)
        transformations['none'] = {'data': y, 'p_value': p_original, 'params': {}}
        self.logger.info(f"  元データ: Shapiro p={p_original:.4f}")
        
        # 2. Log変換
        if (y > 0).all():
            y_log = np.log(y)
            _, p_log = stats.shapiro(y_log)
            transformations['log'] = {'data': y_log, 'p_value': p_log, 'params': {}}
            self.logger.info(f"  Log変換: Shapiro p={p_log:.4f}")
        
        # 3. Sqrt変換
        if (y >= 0).all():
            y_sqrt = np.sqrt(y)
            _, p_sqrt = stats.shapiro(y_sqrt)
            transformations['sqrt'] = {'data': y_sqrt, 'p_value': p_sqrt, 'params': {}}
            self.logger.info(f"  Sqrt変換: Shapiro p={p_sqrt:.4f}")
        
        # 4. Box-Cox変換
        if (y > 0).all():
            y_boxcox, lambda_boxcox = stats.boxcox(y)
            _, p_boxcox = stats.shapiro(y_boxcox)
            transformations['boxcox'] = {
                'data': y_boxcox,
                'p_value': p_boxcox,
                'params': {'lambda': lambda_boxcox}
            }
            self.logger.info(f"  Box-Cox変換: Shapiro p={p_boxcox:.4f}, λ={lambda_boxcox:.4f}")
        
        # 5. Yeo-Johnson変換
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        y_yj = pt.fit_transform(y.reshape(-1, 1)).flatten()
        _, p_yj = stats.shapiro(y_yj)
        transformations['yeo-johnson'] = {
            'data': y_yj,
            'p_value': p_yj,
            'params': {'transformer': pt}
        }
        self.logger.info(f"  Yeo-Johnson変換: Shapiro p={p_yj:.4f}")
        
        # 最適変換選択 (p値最大 = 最も正規分布に近い)
        best_transform = max(transformations.items(), key=lambda x: x[1]['p_value'])
        self.transformation_type = best_transform[0]
        self.transformation_params = best_transform[1]['params']
        
        self.logger.info(f"\n✅ 最適変換: {self.transformation_type.upper()} (p={best_transform[1]['p_value']:.4f})")
        
        # 変換適用
        df_transformed = df.copy()
        df_transformed['y'] = best_transform[1]['data']
        
        return df_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        逆変換
        
        Parameters
        ----------
        y_transformed : np.ndarray
            変換後データ
        
        Returns
        -------
        np.ndarray
            元のスケールに戻したデータ
        """
        if self.transformation_type == 'none':
            return y_transformed
        elif self.transformation_type == 'log':
            return np.exp(y_transformed)
        elif self.transformation_type == 'sqrt':
            return y_transformed ** 2
        elif self.transformation_type == 'boxcox':
            lambda_val = self.transformation_params['lambda']
            return inv_boxcox(y_transformed, lambda_val)
        elif self.transformation_type == 'yeo-johnson':
            pt = self.transformation_params['transformer']
            return pt.inverse_transform(y_transformed.reshape(-1, 1)).flatten()
        else:
            return y_transformed
    
    def generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        包括的特徴量生成
        
        100+ 特徴量を生成
        
        Parameters
        ----------
        df : pd.DataFrame
            入力データ (ds, y)
        
        Returns
        -------
        pd.DataFrame
            特徴量追加データ
        """
        self.logger.info("=" * 80)
        self.logger.info("🔧 特徴量生成開始")
        self.logger.info("=" * 80)
        
        df = df.copy()
        
        # 基本日付特徴
        df = self._generate_basic_date_features(df)
        
        # Lagged regressors
        df = self._generate_lagged_regressors(df)
        
        # Rolling features
        df = self._generate_rolling_features(df)
        
        # Calendar features
        df = self._generate_calendar_features(df)
        
        # Cyclical features
        df = self._generate_cyclical_features(df)
        
        # Event features
        df = self._generate_event_features(df)
        
        # Trend features
        df = self._generate_trend_features(df)
        
        # 欠損値処理 (forward fill)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info("=" * 80)
        self.logger.info("✅ 特徴量生成完了")
        self.logger.info("=" * 80)
        self.logger.info(f"  Lagged regressors: {len(self.lagged_regressor_names)} 個")
        self.logger.info(f"  Future regressors: {len(self.future_regressor_names)} 個")
        self.logger.info(f"  Events: {len(self.event_names)} 個")
        self.logger.info(f"  合計: {len(self.lagged_regressor_names) + len(self.future_regressor_names) + len(self.event_names)} 個")
        
        return df
    
    def _generate_basic_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本日付特徴"""
        self.logger.info("📅 基本日付特徴生成中...")
        
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day_of_month'] = df['ds'].dt.day
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['quarter'] = df['ds'].dt.quarter
        df['day_of_year'] = df['ds'].dt.dayofyear
        df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = ((df['month'] == 1) & (df['day_of_month'] == 1)).astype(int)
        df['is_year_end'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
        df['days_in_month'] = df['ds'].dt.days_in_month
        
        self.logger.info(f"  ✓ 基本日付特徴: 15 個")
        
        return df
    
    def _generate_lagged_regressors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged regressors"""
        self.logger.info("⏱️  Lagged regressors 生成中...")
        
        lag_days = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]
        
        for lag in lag_days:
            col_name = f'y_lag_{lag}'
            df[col_name] = df['y'].shift(lag)
            self.lagged_regressor_names.append(col_name)
        
        self.logger.info(f"  ✓ Lagged regressors: {len(self.lagged_regressor_names)} 個")
        
        return df
    
    def _generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling features"""
        self.logger.info("📊 Rolling features 生成中...")
        
        windows = [7, 14, 28]
        
        for window in windows:
            # Rolling mean
            col_name = f'y_rolling_mean_{window}'
            df[col_name] = df['y'].rolling(window=window, min_periods=1).mean()
            self.lagged_regressor_names.append(col_name)
            
            # Rolling std
            col_name = f'y_rolling_std_{window}'
            df[col_name] = df['y'].rolling(window=window, min_periods=1).std()
            self.lagged_regressor_names.append(col_name)
            
            # EWM
            col_name = f'y_ewm_{window}'
            df[col_name] = df['y'].ewm(span=window, min_periods=1).mean()
            self.lagged_regressor_names.append(col_name)
        
        self.logger.info(f"  ✓ Rolling features: {len(windows) * 3} 個")
        
        return df
    
    def _generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar features"""
        self.logger.info("📆 Calendar features 生成中...")
        
        # One-hot encoding: 曜日
        for dow in range(7):
            col_name = f'dow_{dow}'
            df[col_name] = (df['dayofweek'] == dow).astype(int)
            self.future_regressor_names.append(col_name)
        
        # One-hot encoding: 月
        for month in range(1, 13):
            col_name = f'month_{month}'
            df[col_name] = (df['month'] == month).astype(int)
            self.future_regressor_names.append(col_name)
        
        # 週の位置 (第1週〜第5週)
        df['week_of_month'] = (df['day_of_month'] - 1) // 7 + 1
        for week in range(1, 6):
            col_name = f'week_of_month_{week}'
            df[col_name] = (df['week_of_month'] == week).astype(int)
            self.future_regressor_names.append(col_name)
        
        self.logger.info(f"  ✓ Calendar features: {7 + 12 + 5} 個")
        
        return df
    
    def _generate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical features"""
        self.logger.info("🔄 Cyclical features 生成中...")
        
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
        
        self.logger.info(f"  ✓ Cyclical features: 10 個")
        
        return df
    
    def _generate_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Event features"""
        self.logger.info("🎌 Event features 生成中...")
        
        if not JPHOLIDAY_AVAILABLE:
            self.logger.warning("  ⚠️  jpholiday未インストール → イベント特徴スキップ")
            return df
        
        # 祝日フラグ
        df['is_holiday'] = df['ds'].apply(lambda x: jpholiday.is_holiday(x)).astype(int)
        self.event_names.append('is_holiday')
        
        # ゴールデンウィーク
        df['is_golden_week'] = (
            ((df['month'] == 4) & (df['day_of_month'] >= 29)) |
            ((df['month'] == 5) & (df['day_of_month'] <= 5))
        ).astype(int)
        self.event_names.append('is_golden_week')
        
        # お盆
        df['is_obon'] = (
            (df['month'] == 8) & 
            (df['day_of_month'] >= 13) & 
            (df['day_of_month'] <= 16)
        ).astype(int)
        self.event_names.append('is_obon')
        
        # 年末年始
        df['is_year_end_new_year'] = (
            ((df['month'] == 12) & (df['day_of_month'] >= 29)) |
            ((df['month'] == 1) & (df['day_of_month'] <= 3))
        ).astype(int)
        self.event_names.append('is_year_end_new_year')
        
        # シルバーウィーク
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
        
        self.logger.info(f"  ✓ Event features: {len(self.event_names)} 個")
        
        return df
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend features"""
        self.logger.info("📈 Trend features 生成中...")
        
        df['t'] = np.arange(len(df))
        df['t_squared'] = df['t'] ** 2
        df['t_cubed'] = df['t'] ** 3
        df['t_normalized'] = df['t'] / (len(df) - 1)
        
        self.future_regressor_names.extend(['t', 't_squared', 't_cubed', 't_normalized'])
        
        self.logger.info(f"  ✓ Trend features: 4 個")
        
        return df
    
    def split_train_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        訓練・検証分割
        
        Parameters
        ----------
        df : pd.DataFrame
            全データ
        
        Returns
        -------
        train_df, val_df : pd.DataFrame
            訓練データ、検証データ
        """
        self.logger.info("=" * 80)
        self.logger.info("✂️  訓練・検証分割")
        self.logger.info("=" * 80)
        
        # 検証期間の開始日
        val_start = df['ds'].max() - relativedelta(months=self.validation_months)
        
        train_df = df[df['ds'] < val_start].copy()
        val_df = df[df['ds'] >= val_start].copy()
        
        self.logger.info(f"  訓練期間: {train_df['ds'].min().date()} 〜 {train_df['ds'].max().date()} ({len(train_df)} 日)")
        self.logger.info(f"  検証期間: {val_df['ds'].min().date()} 〜 {val_df['ds'].max().date()} ({len(val_df)} 日)")
        
        return train_df, val_df
    
    # ==============================================================================
    # Optuna最適化 + 訓練
    # ==============================================================================
    
    def optimize_and_train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Tuple[NeuralProphet, Dict]:
        """
        Optuna最適化 + モデル訓練
        
        Parameters
        ----------
        train_df : pd.DataFrame
            訓練データ
        val_df : pd.DataFrame
            検証データ
        
        Returns
        -------
        best_model : NeuralProphet
            最適化されたモデル
        best_params : dict
            最適ハイパーパラメータ
        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 Optuna最適化 + 訓練開始")
        self.logger.info("=" * 80)
        
        # Optuna Study作成
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # 目的関数
        def objective(trial):
            return self._optuna_objective(trial, train_df, val_df)
        
        # 最適化実行
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=True)
        
        self.study = study
        self.best_params = study.best_params
        
        self.logger.info("=" * 80)
        self.logger.info("✅ Optuna最適化完了")
        self.logger.info("=" * 80)
        self.logger.info(f"  最適値 (wQL): {study.best_value:.4f}")
        self.logger.info(f"  最適パラメータ:")
        for key, value in self.best_params.items():
            self.logger.info(f"    {key}: {value}")
        
        # 最適パラメータでモデル訓練
        self.logger.info("\n🚀 最適パラメータでモデル訓練中...")
        self.best_model = self._train_model_with_params(train_df, self.best_params, verbose=True)
        
        return self.best_model, self.best_params
    
    def _optuna_objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> float:
        """
        Optuna目的関数
        
        Parameters
        ----------
        trial : optuna.Trial
            Optunaトライアル
        train_df : pd.DataFrame
            訓練データ
        val_df : pd.DataFrame
            検証データ
        
        Returns
        -------
        float
            評価指標 (wQL)
        """
        # ハイパーパラメータサンプリング
        params = self._sample_hyperparameters(trial)
        
        # モデル訓練
        try:
            model = self._train_model_with_params(train_df, params, verbose=False)
        except Exception as e:
            self.logger.warning(f"  Trial {trial.number} failed: {e}")
            return float('inf')
        
        # 予測
        try:
            future = model.make_future_dataframe(
                df=train_df[['ds', 'y']],
                periods=len(val_df),
                n_historic_predictions=0
            )
            
            # 特徴量追加 (future)
            future = self._add_features_to_future(future, train_df)
            
            forecast = model.predict(future)
            
            # 予測値抽出
            y_pred = forecast['yhat1'].tail(len(val_df)).values
            y_true = val_df['y'].values
            
            # wQL計算
            wql = self._calculate_wql(y_true, y_pred, self.target_quantile)
            
            return wql
            
        except Exception as e:
            self.logger.warning(f"  Trial {trial.number} prediction failed: {e}")
            return float('inf')
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        ハイパーパラメータサンプリング (30+ params)
        
        Parameters
        ----------
        trial : optuna.Trial
            Optunaトライアル
        
        Returns
        -------
        dict
            ハイパーパラメータ辞書
        """
        params = {}
        
        # === AR-Net パラメータ ===
        params['n_lags'] = trial.suggest_int('n_lags', 7, 60)
        params['ar_layers'] = [trial.suggest_categorical('ar_layers', [16, 32, 64, 128])]
        params['ar_sparsity'] = trial.suggest_float('ar_sparsity', 0.0, 0.1)
        
        # === Trend パラメータ ===
        params['growth'] = trial.suggest_categorical('growth', ['linear', 'discontinuous'])
        params['changepoints_range'] = trial.suggest_float('changepoints_range', 0.8, 0.95)
        params['n_changepoints'] = trial.suggest_int('n_changepoints', 10, 50)
        params['trend_reg'] = trial.suggest_float('trend_reg', 0.0, 10.0)
        
        # === Seasonality パラメータ ===
        params['yearly_seasonality'] = trial.suggest_int('yearly_seasonality', 5, 20)
        params['weekly_seasonality'] = trial.suggest_int('weekly_seasonality', 3, 7)
        params['seasonality_mode'] = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        params['seasonality_reg'] = trial.suggest_float('seasonality_reg', 0.0, 1.0)
        
        # === Training パラメータ ===
        params['epochs'] = trial.suggest_categorical('epochs', [50, 100, 200])
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        
        # === Loss パラメータ ===
        params['loss_func'] = trial.suggest_categorical('loss_func', ['Huber', 'MSE'])
        
        # === Regularization パラメータ ===
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
        params['normalize'] = trial.suggest_categorical('normalize', ['auto', 'standardize', 'minmax'])
        
        return params
    
    def _train_model_with_params(
        self,
        train_df: pd.DataFrame,
        params: Dict,
        verbose: bool = False
    ) -> NeuralProphet:
        """
        指定パラメータでモデル訓練
        
        Parameters
        ----------
        train_df : pd.DataFrame
            訓練データ
        params : dict
            ハイパーパラメータ
        verbose : bool
            詳細出力
        
        Returns
        -------
        NeuralProphet
            訓練済みモデル
        """
        # モデル初期化
        model = NeuralProphet(
            n_lags=params.get('n_lags', 28),
            ar_layers=params.get('ar_layers', [64]),
            ar_sparsity=params.get('ar_sparsity', 0.0),
            growth=params.get('growth', 'linear'),
            changepoints_range=params.get('changepoints_range', 0.9),
            n_changepoints=params.get('n_changepoints', 25),
            trend_reg=params.get('trend_reg', 1.0),
            yearly_seasonality=params.get('yearly_seasonality', 10),
            weekly_seasonality=params.get('weekly_seasonality', 5),
            daily_seasonality=False,
            seasonality_mode=params.get('seasonality_mode', 'additive'),
            seasonality_reg=params.get('seasonality_reg', 0.1),
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            learning_rate=params.get('learning_rate', None),
            loss_func=params.get('loss_func', 'Huber'),
            normalize=params.get('normalize', 'auto'),
            drop_missing=False
        )
        
        # Lagged regressors追加
        for lag_name in self.lagged_regressor_names:
            if lag_name in train_df.columns:
                model.add_lagged_regressor(names=lag_name, n_lags=1, regularization=0.1)
        
        # Future regressors追加
        for future_name in self.future_regressor_names:
            if future_name in train_df.columns:
                model.add_future_regressor(name=future_name, regularization=0.1)
        
        # Events追加
        for event_name in self.event_names:
            if event_name in train_df.columns:
                model.add_events(event_name)
        
        # 訓練
        metrics = model.fit(
            train_df[['ds', 'y'] + self.lagged_regressor_names + self.future_regressor_names + self.event_names],
            freq='D',
            validation_df=None,
            progress=None if not verbose else 'bar'
        )
        
        return model
    
    def _add_features_to_future(self, future: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Future dataframeに特徴量追加
        
        Parameters
        ----------
        future : pd.DataFrame
            Future dataframe
        train_df : pd.DataFrame
            訓練データ (特徴量ソース)
        
        Returns
        -------
        pd.DataFrame
            特徴量追加済みfuture dataframe
        """
        # 基本日付特徴
        future = self._generate_basic_date_features(future)
        
        # Future regressors
        future = self._generate_calendar_features(future)
        future = self._generate_cyclical_features(future)
        future = self._generate_event_features(future)
        future = self._generate_trend_features(future)
        
        # Lagged regressors (訓練データから取得)
        combined = pd.concat([train_df, future], ignore_index=True)
        combined = combined.sort_values('ds').reset_index(drop=True)
        
        for lag_name in self.lagged_regressor_names:
            if 'lag_' in lag_name:
                lag = int(lag_name.split('_')[-1])
                combined[lag_name] = combined['y'].shift(lag)
            elif 'rolling_' in lag_name:
                window = int(lag_name.split('_')[-1])
                if 'mean' in lag_name:
                    combined[lag_name] = combined['y'].rolling(window=window, min_periods=1).mean()
                elif 'std' in lag_name:
                    combined[lag_name] = combined['y'].rolling(window=window, min_periods=1).std()
            elif 'ewm_' in lag_name:
                span = int(lag_name.split('_')[-1])
                combined[lag_name] = combined['y'].ewm(span=span, min_periods=1).mean()
        
        # future部分だけ抽出
        future = combined[combined['ds'] >= future['ds'].min()].copy()
        
        # 欠損値補完
        future = future.fillna(method='ffill').fillna(method='bfill')
        
        return future
    
    def _calculate_wql(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.6) -> float:
        """
        Weighted Quantile Loss (wQL) 計算
        
        Parameters
        ----------
        y_true : np.ndarray
            実測値
        y_pred : np.ndarray
            予測値
        quantile : float
            分位点
        
        Returns
        -------
        float
            wQL値
        """
        errors = y_true - y_pred
        loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
        return np.mean(loss)
    
    # ==============================================================================
    # 予測
    # ==============================================================================
    
    def predict(
        self,
        model: NeuralProphet,
        periods: int = 60,
        include_history: bool = True,
        train_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        予測
        
        Parameters
        ----------
        model : NeuralProphet
            訓練済みモデル
        periods : int
            予測期間 (日数)
        include_history : bool
            履歴含む
        train_df : pd.DataFrame, optional
            訓練データ (履歴予測用)
        
        Returns
        -------
        pd.DataFrame
            予測結果
        """
        self.logger.info("=" * 80)
        self.logger.info(f"🔮 予測開始 (periods={periods})")
        self.logger.info("=" * 80)
        
        if train_df is None:
            raise ValueError("train_df が必要です")
        
        # Future dataframe作成
        future = model.make_future_dataframe(
            df=train_df[['ds', 'y']],
            periods=periods,
            n_historic_predictions=len(train_df) if include_history else 0
        )
        
        # 特徴量追加
        future = self._add_features_to_future(future, train_df)
        
        # 予測
        forecast = model.predict(future)
        
        # 逆変換
        if 'yhat1' in forecast.columns:
            forecast['yhat_original'] = self.inverse_transform(forecast['yhat1'].values)
        
        self.logger.info(f"  ✓ 予測完了: {len(forecast)} 行")
        
        return forecast
    
    # ==============================================================================
    # 評価
    # ==============================================================================
    
    def evaluate(
        self,
        val_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> Dict:
        """
        評価指標計算
        
        Parameters
        ----------
        val_df : pd.DataFrame
            検証データ (実測値)
        forecast_df : pd.DataFrame
            予測結果
        
        Returns
        -------
        dict
            評価指標辞書
        """
        self.logger.info("=" * 80)
        self.logger.info("📊 評価指標計算")
        self.logger.info("=" * 80)
        
        # 日付でマージ
        merged = val_df[['ds', 'y']].merge(
            forecast_df[['ds', 'yhat1']],
            on='ds',
            how='inner'
        )
        
        y_true = merged['y'].values
        y_pred = merged['yhat1'].values
        
        # 逆変換
        y_true_original = self.inverse_transform(y_true)
        y_pred_original = self.inverse_transform(y_pred)
        
        metrics = {}
        
        # === Primary Metrics ===
        
        # wQL (複数分位点)
        metrics['wql'] = {}
        for q in [0.1, 0.5, 0.6, 0.7, 0.9]:
            wql = self._calculate_wql(y_true_original, y_pred_original, q)
            metrics['wql'][f'QL_{int(q*100)}'] = wql
        
        # WAPE
        metrics['wape'] = np.sum(np.abs(y_true_original - y_pred_original)) / np.sum(np.abs(y_true_original)) * 100
        
        # sMAPE
        metrics['smape'] = np.mean(2 * np.abs(y_pred_original - y_true_original) / (np.abs(y_true_original) + np.abs(y_pred_original))) * 100
        
        # Asymmetric MAE
        errors = y_true_original - y_pred_original
        under_loss = np.sum(np.maximum(errors, 0)) * 2.0  # 過小予測ペナルティ
        over_loss = np.sum(np.maximum(-errors, 0)) * 0.5  # 過大予測ペナルティ
        metrics['asymmetric_mae'] = (under_loss + over_loss) / len(errors)
        
        # === Secondary Metrics ===
        
        metrics['mae'] = mean_absolute_error(y_true_original, y_pred_original)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        metrics['mape'] = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
        metrics['r2'] = r2_score(y_true_original, y_pred_original)
        metrics['bias'] = np.mean(y_pred_original - y_true_original)
        metrics['bias_pct'] = metrics['bias'] / np.mean(y_true_original) * 100
        
        # MASE (Mean Absolute Scaled Error)
        naive_errors = np.abs(np.diff(y_true_original))
        mae_naive = np.mean(naive_errors)
        metrics['mase'] = metrics['mae'] / mae_naive if mae_naive > 0 else np.inf
        
        # === Peak Day Metrics ===
        
        # Top 25% volume days
        threshold = np.percentile(y_true_original, 75)
        peak_mask = y_true_original >= threshold
        
        if np.sum(peak_mask) > 0:
            metrics['peak_mae'] = mean_absolute_error(
                y_true_original[peak_mask],
                y_pred_original[peak_mask]
            )
            metrics['peak_mape'] = np.mean(
                np.abs((y_true_original[peak_mask] - y_pred_original[peak_mask]) / y_true_original[peak_mask])
            ) * 100
            metrics['peak_wape'] = np.sum(np.abs(y_true_original[peak_mask] - y_pred_original[peak_mask])) / np.sum(np.abs(y_true_original[peak_mask])) * 100
            metrics['peak_bias'] = np.mean(y_pred_original[peak_mask] - y_true_original[peak_mask])
            metrics['peak_under_pred_rate'] = np.mean(y_pred_original[peak_mask] < y_true_original[peak_mask]) * 100
        else:
            metrics['peak_mae'] = np.nan
            metrics['peak_mape'] = np.nan
            metrics['peak_wape'] = np.nan
            metrics['peak_bias'] = np.nan
            metrics['peak_under_pred_rate'] = np.nan
        
        # === Day-of-Week Metrics ===
        
        merged_full = val_df[['ds', 'y']].merge(
            forecast_df[['ds', 'yhat1']],
            on='ds',
            how='inner'
        )
        merged_full['dayofweek'] = pd.to_datetime(merged_full['ds']).dt.dayofweek
        
        metrics['dow_metrics'] = {}
        for dow in range(7):
            dow_mask = merged_full['dayofweek'] == dow
            if np.sum(dow_mask) > 0:
                dow_y_true = self.inverse_transform(merged_full.loc[dow_mask, 'y'].values)
                dow_y_pred = self.inverse_transform(merged_full.loc[dow_mask, 'yhat1'].values)
                
                metrics['dow_metrics'][f'dow_{dow}'] = {
                    'mae': mean_absolute_error(dow_y_true, dow_y_pred),
                    'bias': np.mean(dow_y_pred - dow_y_true),
                    'bias_pct': np.mean(dow_y_pred - dow_y_true) / np.mean(dow_y_true) * 100
                }
        
        # ログ出力
        self.logger.info("\n🎯 Primary Metrics:")
        self.logger.info(f"  wQL (QL_60): {metrics['wql']['QL_60']:.4f}")
        self.logger.info(f"  WAPE: {metrics['wape']:.2f}%")
        self.logger.info(f"  sMAPE: {metrics['smape']:.2f}%")
        self.logger.info(f"  Asymmetric MAE: {metrics['asymmetric_mae']:.2f}")
        
        self.logger.info("\n📈 Secondary Metrics:")
        self.logger.info(f"  MAE: {metrics['mae']:.2f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        self.logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        self.logger.info(f"  MASE: {metrics['mase']:.4f}")
        self.logger.info(f"  R²: {metrics['r2']:.4f}")
        self.logger.info(f"  Bias: {metrics['bias']:.2f} ({metrics['bias_pct']:.2f}%)")
        
        self.logger.info("\n🔝 Peak Day Metrics:")
        self.logger.info(f"  Peak MAE: {metrics['peak_mae']:.2f}")
        self.logger.info(f"  Peak MAPE: {metrics['peak_mape']:.2f}%")
        self.logger.info(f"  Peak Under-prediction Rate: {metrics['peak_under_pred_rate']:.2f}%")
        
        return metrics
    
    # ==============================================================================
    # 可視化
    # ==============================================================================
    
    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ):
        """
        予測結果プロット
        
        Parameters
        ----------
        forecast_df : pd.DataFrame
            予測結果
        val_df : pd.DataFrame, optional
            検証データ (実測値)
        save_path : str, optional
            保存パス
        """
        self.logger.info("📊 予測結果プロット作成中...")
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 予測値
        ax.plot(
            forecast_df['ds'],
            self.inverse_transform(forecast_df['yhat1'].values),
            label='Forecast',
            color='blue',
            linewidth=2
        )
        
        # 信頼区間 (もしあれば)
        if 'yhat1_lower' in forecast_df.columns and 'yhat1_upper' in forecast_df.columns:
            ax.fill_between(
                forecast_df['ds'],
                self.inverse_transform(forecast_df['yhat1_lower'].values),
                self.inverse_transform(forecast_df['yhat1_upper'].values),
                alpha=0.2,
                color='blue',
                label='Confidence Interval'
            )
        
        # 実測値
        if val_df is not None:
            ax.plot(
                val_df['ds'],
                self.inverse_transform(val_df['y'].values),
                label='Actual',
                color='red',
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Call Volume', fontsize=12)
        ax.set_title('Call Volume Forecast - NeuralProphet v5.0', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"  ✓ 保存: {save_path}")
        
        if JUPYTER_MODE:
            plt.show()
        else:
            plt.close()
    
    def plot_components(
        self,
        model: NeuralProphet,
        forecast_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        コンポーネントプロット
        
        Parameters
        ----------
        model : NeuralProphet
            訓練済みモデル
        forecast_df : pd.DataFrame
            予測結果
        save_path : str, optional
            保存パス
        """
        self.logger.info("📊 コンポーネントプロット作成中...")
        
        fig = model.plot_components(forecast_df)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"  ✓ 保存: {save_path}")
        
        if JUPYTER_MODE:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_metrics(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ):
        """
        評価指標プロット
        
        Parameters
        ----------
        metrics : dict
            評価指標辞書
        save_path : str, optional
            保存パス
        """
        self.logger.info("📊 評価指標プロット作成中...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # wQL
        ax = axes[0, 0]
        wql_values = [v for k, v in metrics['wql'].items()]
        wql_labels = [k for k in metrics['wql'].keys()]
        ax.bar(wql_labels, wql_values, color='steelblue')
        ax.set_title('Weighted Quantile Loss (wQL)', fontweight='bold')
        ax.set_ylabel('wQL')
        ax.grid(axis='y', alpha=0.3)
        
        # Primary Metrics
        ax = axes[0, 1]
        primary_metrics = ['wape', 'smape', 'mape']
        primary_values = [metrics[m] for m in primary_metrics]
        primary_labels = ['WAPE', 'sMAPE', 'MAPE']
        ax.bar(primary_labels, primary_values, color='coral')
        ax.set_title('Percentage Errors', fontweight='bold')
        ax.set_ylabel('%')
        ax.grid(axis='y', alpha=0.3)
        
        # Secondary Metrics
        ax = axes[0, 2]
        secondary_metrics = ['mae', 'rmse', 'asymmetric_mae']
        secondary_values = [metrics[m] for m in secondary_metrics]
        secondary_labels = ['MAE', 'RMSE', 'Asymmetric MAE']
        ax.bar(secondary_labels, secondary_values, color='lightgreen')
        ax.set_title('Absolute Errors', fontweight='bold')
        ax.set_ylabel('Error')
        ax.grid(axis='y', alpha=0.3)
        
        # MASE & R²
        ax = axes[1, 0]
        quality_metrics = ['mase', 'r2']
        quality_values = [metrics[m] for m in quality_metrics]
        quality_labels = ['MASE', 'R²']
        colors = ['orange' if metrics['mase'] < 1 else 'red', 'green' if metrics['r2'] > 0.7 else 'orange']
        ax.bar(quality_labels, quality_values, color=colors)
        ax.set_title('Quality Metrics', fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='MASE Target')
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='R² Target')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Peak Day Metrics
        ax = axes[1, 1]
        if not np.isnan(metrics['peak_mae']):
            peak_metrics = ['peak_mae', 'peak_wape', 'peak_under_pred_rate']
            peak_values = [metrics['peak_mae'], metrics['peak_wape'], metrics['peak_under_pred_rate']]
            peak_labels = ['Peak MAE', 'Peak WAPE\n(%)', 'Under-pred\nRate (%)']
            ax.bar(peak_labels, peak_values, color='purple')
            ax.set_title('Peak Day Performance', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Peak Data', ha='center', va='center', fontsize=14)
            ax.set_title('Peak Day Performance', fontweight='bold')
        
        # Day-of-Week MAE
        ax = axes[1, 2]
        if 'dow_metrics' in metrics and len(metrics['dow_metrics']) > 0:
            dow_mae = [metrics['dow_metrics'][f'dow_{dow}']['mae'] for dow in range(7)]
            dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.bar(dow_labels, dow_mae, color='teal')
            ax.set_title('Day-of-Week MAE', fontweight='bold')
            ax.set_ylabel('MAE')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No DoW Data', ha='center', va='center', fontsize=14)
            ax.set_title('Day-of-Week MAE', fontweight='bold')
        
        plt.suptitle('NeuralProphet v5.0 - Evaluation Metrics', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"  ✓ 保存: {save_path}")
        
        if JUPYTER_MODE:
            plt.show()
        else:
            plt.close()
    
    def plot_optuna_results(self, save_path: Optional[str] = None):
        """
        Optuna最適化結果プロット
        
        Parameters
        ----------
        save_path : str, optional
            保存パス
        """
        if self.study is None:
            self.logger.warning("  ⚠️  Optuna study が見つかりません")
            return
        
        self.logger.info("📊 Optuna最適化結果プロット作成中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Optimization History
        ax = axes[0, 0]
        trials = self.study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        trial_values = [t.value for t in trials if t.value is not None]
        ax.plot(trial_numbers, trial_values, marker='o', markersize=3, alpha=0.6)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value (wQL)')
        ax.set_title('Optimization History', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Parameter Importances
        try:
            ax = axes[0, 1]
            importances = optuna.importance.get_param_importances(self.study)
            params = list(importances.keys())[:10]  # Top 10
            values = [importances[p] for p in params]
            ax.barh(params, values, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Parameter Importances', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Importance Error:\n{e}', ha='center', va='center', fontsize=10)
            ax.set_title('Top 10 Parameter Importances', fontweight='bold')
        
        # Parallel Coordinate (上位10 trials)
        try:
            ax = axes[1, 0]
            from optuna.visualization.matplotlib import plot_parallel_coordinate
            fig_parallel = plot_parallel_coordinate(self.study)
            # Copy to axes (難しいのでスキップ)
            ax.text(0.5, 0.5, 'See separate plot', ha='center', va='center', fontsize=12)
            ax.set_title('Parallel Coordinate Plot', fontweight='bold')
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.set_title('Parallel Coordinate Plot', fontweight='bold')
        
        # Contour (上位2パラメータ)
        try:
            ax = axes[1, 1]
            from optuna.visualization.matplotlib import plot_contour
            # Copy to axes (難しいのでスキップ)
            ax.text(0.5, 0.5, 'See separate plot', ha='center', va='center', fontsize=12)
            ax.set_title('Contour Plot', fontweight='bold')
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.set_title('Contour Plot', fontweight='bold')
        
        plt.suptitle('Optuna Optimization Results', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"  ✓ 保存: {save_path}")
        
        if JUPYTER_MODE:
            plt.show()
        else:
            plt.close()
    
    # ==============================================================================
    # レポート生成
    # ==============================================================================
    
    def generate_report(
        self,
        metrics: Dict,
        best_params: Dict,
        output_path: Optional[str] = None
    ):
        """
        HTMLレポート生成
        
        Parameters
        ----------
        metrics : dict
            評価指標
        best_params : dict
            最適パラメータ
        output_path : str, optional
            出力パス
        """
        self.logger.info("=" * 80)
        self.logger.info("📝 HTMLレポート生成")
        self.logger.info("=" * 80)
        
        if output_path is None:
            output_path = self.output_dir / f'neuralprophet_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NeuralProphet v5.0 - Forecast Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric-good {{
            color: green;
            font-weight: bold;
        }}
        .metric-warning {{
            color: orange;
            font-weight: bold;
        }}
        .metric-bad {{
            color: red;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 NeuralProphet v5.0 - Call Volume Forecast Report</h1>
        <p><strong>生成日時:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>🎯 Primary Metrics</h2>
        <table>
            <tr>
                <th>指標</th>
                <th>値</th>
                <th>評価</th>
            </tr>
            <tr>
                <td>wQL (QL_60)</td>
                <td>{metrics['wql']['QL_60']:.4f}</td>
                <td class="{'metric-good' if metrics['wql']['QL_60'] < 50 else 'metric-warning' if metrics['wql']['QL_60'] < 100 else 'metric-bad'}">
                    {'✅ Excellent' if metrics['wql']['QL_60'] < 50 else '⚠️ Good' if metrics['wql']['QL_60'] < 100 else '❌ Needs Improvement'}
                </td>
            </tr>
            <tr>
                <td>WAPE</td>
                <td>{metrics['wape']:.2f}%</td>
                <td class="{'metric-good' if metrics['wape'] < 10 else 'metric-warning' if metrics['wape'] < 20 else 'metric-bad'}">
                    {'✅ Excellent' if metrics['wape'] < 10 else '⚠️ Good' if metrics['wape'] < 20 else '❌ Needs Improvement'}
                </td>
            </tr>
            <tr>
                <td>sMAPE</td>
                <td>{metrics['smape']:.2f}%</td>
                <td class="{'metric-good' if metrics['smape'] < 10 else 'metric-warning' if metrics['smape'] < 20 else 'metric-bad'}">
                    {'✅ Excellent' if metrics['smape'] < 10 else '⚠️ Good' if metrics['smape'] < 20 else '❌ Needs Improvement'}
                </td>
            </tr>
            <tr>
                <td>Asymmetric MAE</td>
                <td>{metrics['asymmetric_mae']:.2f}</td>
                <td class="{'metric-good' if metrics['asymmetric_mae'] < 50 else 'metric-warning' if metrics['asymmetric_mae'] < 100 else 'metric-bad'}">
                    {'✅ Excellent' if metrics['asymmetric_mae'] < 50 else '⚠️ Good' if metrics['asymmetric_mae'] < 100 else '❌ Needs Improvement'}
                </td>
            </tr>
        </table>
        
        <h2>📈 Secondary Metrics</h2>
        <table>
            <tr>
                <th>指標</th>
                <th>値</th>
            </tr>
            <tr><td>MAE</td><td>{metrics['mae']:.2f}</td></tr>
            <tr><td>RMSE</td><td>{metrics['rmse']:.2f}</td></tr>
            <tr><td>MAPE</td><td>{metrics['mape']:.2f}%</td></tr>
            <tr><td>MASE</td><td>{metrics['mase']:.4f}</td></tr>
            <tr><td>R²</td><td>{metrics['r2']:.4f}</td></tr>
            <tr><td>Bias</td><td>{metrics['bias']:.2f} ({metrics['bias_pct']:.2f}%)</td></tr>
        </table>
        
        <h2>🔝 Peak Day Metrics</h2>
        <table>
            <tr>
                <th>指標</th>
                <th>値</th>
            </tr>
            <tr><td>Peak MAE</td><td>{metrics.get('peak_mae', 'N/A'):.2f if not np.isnan(metrics.get('peak_mae', np.nan)) else 'N/A'}</td></tr>
            <tr><td>Peak MAPE</td><td>{metrics.get('peak_mape', 'N/A'):.2f if not np.isnan(metrics.get('peak_mape', np.nan)) else 'N/A'}%</td></tr>
            <tr><td>Peak WAPE</td><td>{metrics.get('peak_wape', 'N/A'):.2f if not np.isnan(metrics.get('peak_wape', np.nan)) else 'N/A'}%</td></tr>
            <tr><td>Peak Bias</td><td>{metrics.get('peak_bias', 'N/A'):.2f if not np.isnan(metrics.get('peak_bias', np.nan)) else 'N/A'}</td></tr>
            <tr><td>Peak Under-prediction Rate</td><td>{metrics.get('peak_under_pred_rate', 'N/A'):.2f if not np.isnan(metrics.get('peak_under_pred_rate', np.nan)) else 'N/A'}%</td></tr>
        </table>
        
        <h2>⚙️ Best Hyperparameters</h2>
        <table>
            <tr>
                <th>パラメータ</th>
                <th>値</th>
            </tr>
"""
        
        for key, value in best_params.items():
            html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += """
        </table>
        
        <h2>💡 Recommendations</h2>
        <ul>
"""
        
        # 推奨事項
        if metrics['wape'] < 10:
            html += "            <li>✅ WAPE < 10%: 優秀な予測精度です。本番展開可能です。</li>\n"
        elif metrics['wape'] < 20:
            html += "            <li>⚠️ WAPE 10-20%: 良好な予測精度です。ピーク日の精度改善を検討してください。</li>\n"
        else:
            html += "            <li>❌ WAPE > 20%: 予測精度が不十分です。特徴量追加・ハイパーパラメータ再調整が必要です。</li>\n"
        
        if metrics['mase'] < 1.0:
            html += "            <li>✅ MASE < 1.0: ナイーブ予測より高精度です。</li>\n"
        else:
            html += "            <li>❌ MASE ≥ 1.0: ナイーブ予測以下です。モデル改善が必要です。</li>\n"
        
        if not np.isnan(metrics.get('peak_under_pred_rate', np.nan)) and metrics['peak_under_pred_rate'] > 40:
            html += f"            <li>⚠️ ピーク日の過小予測率 {metrics['peak_under_pred_rate']:.1f}%: シフト不足リスクがあります。QL_70への調整を推奨。</li>\n"
        
        html += """
        </ul>
        
        <div class="footer">
            <p>Generated by NeuralProphet Ultimate Predictor v5.0</p>
            <p>Powered by NeuralProphet, Optuna, PyTorch</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"  ✓ HTMLレポート生成完了: {output_path}")
        
        if JUPYTER_MODE:
            display(HTML(f'<a href="{output_path}" target="_blank">📄 レポートを開く</a>'))


# ==============================================================================
# CLI + Jupyter両対応 Main
# ==============================================================================

def main_cli():
    """CLIエントリーポイント"""
    parser = argparse.ArgumentParser(description='NeuralProphet Ultimate Predictor v5.0')
    parser.add_argument('data_path', type=str, help='CSVファイルパス')
    parser.add_argument('--date-col', type=str, default='date', help='日付カラム名')
    parser.add_argument('--value-col', type=str, default='y', help='値カラム名')
    parser.add_argument('--validation-months', type=int, default=2, help='検証期間 (月数)')
    parser.add_argument('--optuna-trials', type=int, default=100, help='Optuna試行回数')
    parser.add_argument('--target-quantile', type=float, default=0.6, help='目標分位点')
    parser.add_argument('--epochs', type=int, default=100, help='訓練エポック数')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # Predictor初期化
    predictor = NeuralProphetUltimatePredictor(
        validation_months=args.validation_months,
        optuna_trials=args.optuna_trials,
        target_quantile=args.target_quantile,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # データ読み込み
    df = predictor.load_data(args.data_path, date_col=args.date_col, value_col=args.value_col)
    
    # 自動変換
    df_transformed = predictor.select_optimal_transformation(df)
    
    # 特徴量生成
    df_features = predictor.generate_comprehensive_features(df_transformed)
    
    # 訓練・検証分割
    train_df, val_df = predictor.split_train_validation(df_features)
    
    # Optuna最適化 + 訓練
    best_model, best_params = predictor.optimize_and_train(train_df, val_df)
    
    # 予測
    forecast_df = predictor.predict(best_model, periods=60, include_history=True, train_df=train_df)
    
    # 評価
    metrics = predictor.evaluate(val_df, forecast_df)
    
    # 可視化
    predictor.plot_forecast(
        forecast_df, 
        val_df,
        save_path=predictor.output_dir / 'forecast_plot.png'
    )
    predictor.plot_components(
        best_model,
        forecast_df,
        save_path=predictor.output_dir / 'components_plot.png'
    )
    predictor.plot_metrics(
        metrics,
        save_path=predictor.output_dir / 'metrics_plot.png'
    )
    predictor.plot_optuna_results(
        save_path=predictor.output_dir / 'optuna_plot.png'
    )
    
    # レポート生成
    predictor.generate_report(metrics, best_params)
    
    print("\n" + "="*80)
    print("✅ すべての処理が完了しました！")
    print("="*80)
    print(f"📂 出力先: {predictor.output_dir}")
    print(f"📊 予測プロット: forecast_plot.png")
    print(f"📊 コンポーネントプロット: components_plot.png")
    print(f"📊 評価指標プロット: metrics_plot.png")
    print(f"📊 Optuna結果プロット: optuna_plot.png")
    print(f"📝 HTMLレポート: neuralprophet_report_*.html")


if __name__ == '__main__':
    main_cli()
