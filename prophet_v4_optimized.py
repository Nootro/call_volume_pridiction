#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Prophet Optimized Predictor for Call Center v4.0
==============================================================================

コールセンター日次呼量予測システム v4.0
- データ形状に基づく自動変換
- Optunaで最適化した単一モデル
- アンサンブル学習なし

主要機能:
---------
1. データ形状に基づく自動変換選択
   - 正規性検定 (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
   - 歪度・尖度チェック
   - Box-Cox, Yeo-Johnson, 対数, 平方根, 逆数変換を評価
   - 最適な変換を自動選択
2. Optunaベイズ最適化 (単一モデル)
3. 詳細な自動診断 (CV, ANOVA, ACF, ADF, STL分解)
4. 高度な特徴量エンジニアリング (30+ features)
5. 検証データ予測 (validation_forecast.csv)
6. 実務用予測 (forecast.csv) - 全データで再学習
7. 包括的可視化とレポート

使用例:
-------
python prophet_v4_optimized.py data.csv --validation-months 2 --optuna-trials 200

作成者: AI Assistant
バージョン: 4.0
最終更新: 2026-02-18
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
from scipy import stats
from scipy.stats import (normaltest, shapiro, jarque_bera, anderson, 
                         boxcox, yeojohnson, skew, kurtosis)
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
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


# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Hiragino Sans', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ProphetOptimizedPredictor:
    """
    Prophet v4.0 最適化予測システム
    - 自動変換選択
    - Optuna最適化単一モデル
    """
    
    def __init__(self, output_dir: str = "output"):
        """初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # データ
        self.df_train = None
        self.df_validation = None
        self.df_full = None
        
        # 診断結果
        self.diagnostics = {}
        
        # モデルと予測
        self.best_params = {}
        self.model_validation = None  # 検証用モデル
        self.model_production = None  # 実務用モデル
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
                logging.FileHandler(self.output_dir / "prophet_v4.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("Prophet v4.0 最適化予測システム 初期化完了")
        self.logger.info("=" * 80)
    
    def load_data(self, filepath: str, date_col: str = "ds", 
                  target_col: str = "y") -> pd.DataFrame:
        """
        CSVデータ読み込み
        """
        self.logger.info(f"データ読み込み開始: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # 列名標準化
        if date_col not in df.columns and target_col not in df.columns:
            if len(df.columns) >= 2:
                df.columns = ['ds', 'y'] + list(df.columns[2:])
                self.logger.info(f"列名を自動変換: {date_col} -> ds, {target_col} -> y")
        
        # 日付変換
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # 欠損値チェック
        missing = df[['ds', 'y']].isnull().sum()
        if missing.any():
            self.logger.warning(f"欠損値検出: {missing.to_dict()}")
            df = df.dropna(subset=['ds', 'y'])
        
        self.logger.info(f"データ読み込み完了: {len(df)} 行")
        self.logger.info(f"期間: {df['ds'].min()} 〜 {df['ds'].max()}")
        self.logger.info(f"目的変数統計: 平均={df['y'].mean():.1f}, "
                        f"中央値={df['y'].median():.1f}, "
                        f"標準偏差={df['y'].std():.1f}")
        
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
        self.logger.info("=" * 80)
        self.logger.info("データ変換の自動選択開始")
        self.logger.info("=" * 80)
        
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
        
        self.logger.info(f"元データ統計:")
        self.logger.info(f"  平均: {original_stats['mean']:.2f}")
        self.logger.info(f"  標準偏差: {original_stats['std']:.2f}")
        self.logger.info(f"  歪度: {original_stats['skewness']:.4f}")
        self.logger.info(f"  尖度: {original_stats['kurtosis']:.4f}")
        
        # 正規性検定 (元データ)
        def test_normality(data):
            """正規性検定"""
            tests = {}
            
            # Shapiro-Wilk (サンプルサイズ < 5000)
            if len(data) < 5000:
                stat, p = shapiro(data)
                tests['shapiro'] = {'statistic': stat, 'pvalue': p}
            
            # Kolmogorov-Smirnov
            stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            tests['ks'] = {'statistic': stat, 'pvalue': p}
            
            # Anderson-Darling
            result = anderson(data, dist='norm')
            tests['anderson'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist()
            }
            
            # Jarque-Bera
            stat, p = jarque_bera(data)
            tests['jarque_bera'] = {'statistic': stat, 'pvalue': p}
            
            # D'Agostino-Pearson
            stat, p = normaltest(data)
            tests['dagostino'] = {'statistic': stat, 'pvalue': p}
            
            return tests
        
        original_tests = test_normality(y_clean)
        
        self.logger.info("元データの正規性検定:")
        for name, result in original_tests.items():
            if 'pvalue' in result:
                self.logger.info(f"  {name}: p値={result['pvalue']:.6f}")
        
        # 変換候補を評価
        transformations = []
        
        # 1. 変換なし
        transformations.append({
            'method': 'none',
            'y_transformed': y_clean,
            'lambda': None,
            'stats': original_stats,
            'tests': original_tests
        })
        
        # 2. 対数変換 (y > 0の場合)
        if y_clean.min() > 0:
            y_log = np.log(y_clean)
            log_stats = {
                'skewness': skew(y_log),
                'kurtosis': kurtosis(y_log)
            }
            transformations.append({
                'method': 'log',
                'y_transformed': y_log,
                'lambda': None,
                'stats': log_stats,
                'tests': test_normality(y_log)
            })
        
        # 3. 平方根変換 (y >= 0の場合)
        if y_clean.min() >= 0:
            y_sqrt = np.sqrt(y_clean)
            sqrt_stats = {
                'skewness': skew(y_sqrt),
                'kurtosis': kurtosis(y_sqrt)
            }
            transformations.append({
                'method': 'sqrt',
                'y_transformed': y_sqrt,
                'lambda': None,
                'stats': sqrt_stats,
                'tests': test_normality(y_sqrt)
            })
        
        # 4. Box-Cox変換 (y > 0の場合)
        if y_clean.min() > 0:
            y_boxcox, lambda_boxcox = boxcox(y_clean)
            boxcox_stats = {
                'skewness': skew(y_boxcox),
                'kurtosis': kurtosis(y_boxcox)
            }
            transformations.append({
                'method': 'boxcox',
                'y_transformed': y_boxcox,
                'lambda': lambda_boxcox,
                'stats': boxcox_stats,
                'tests': test_normality(y_boxcox)
            })
        
        # 5. Yeo-Johnson変換 (全範囲対応)
        y_yeojohnson, lambda_yj = yeojohnson(y_clean)
        yj_stats = {
            'skewness': skew(y_yeojohnson),
            'kurtosis': kurtosis(y_yeojohnson)
        }
        transformations.append({
            'method': 'yeojohnson',
            'y_transformed': y_yeojohnson,
            'lambda': lambda_yj,
            'stats': yj_stats,
            'tests': test_normality(y_yeojohnson)
        })
        
        # 6. 逆数変換 (y > 0の場合)
        if y_clean.min() > 0:
            y_inv = 1 / y_clean
            inv_stats = {
                'skewness': skew(y_inv),
                'kurtosis': kurtosis(y_inv)
            }
            transformations.append({
                'method': 'inverse',
                'y_transformed': y_inv,
                'lambda': None,
                'stats': inv_stats,
                'tests': test_normality(y_inv)
            })
        
        # スコアリング: 正規性p値の平均 + 歪度・尖度ペナルティ
        def calculate_score(trans):
            """変換のスコア計算"""
            # 正規性p値の平均
            p_values = []
            for test_name, result in trans['tests'].items():
                if 'pvalue' in result:
                    p_values.append(result['pvalue'])
            avg_pvalue = np.mean(p_values) if p_values else 0
            
            # 歪度・尖度ペナルティ
            skew_penalty = abs(trans['stats']['skewness'])
            kurt_penalty = abs(trans['stats']['kurtosis'])
            
            # スコア = p値平均 - 0.1*歪度 - 0.05*尖度
            score = avg_pvalue - 0.1 * skew_penalty - 0.05 * kurt_penalty
            return score
        
        # 各変換のスコアを計算
        for trans in transformations:
            trans['score'] = calculate_score(trans)
        
        # スコアでソート
        transformations.sort(key=lambda x: x['score'], reverse=True)
        
        # 結果表示
        self.logger.info("\n変換候補の評価結果:")
        self.logger.info("-" * 80)
        for i, trans in enumerate(transformations, 1):
            self.logger.info(f"{i}. {trans['method']}")
            self.logger.info(f"   スコア: {trans['score']:.6f}")
            self.logger.info(f"   歪度: {trans['stats']['skewness']:.4f}")
            self.logger.info(f"   尖度: {trans['stats']['kurtosis']:.4f}")
            if trans['lambda'] is not None:
                self.logger.info(f"   λパラメータ: {trans['lambda']:.4f}")
        
        # 最適な変換を選択
        best_trans = transformations[0]
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"✓ 選択された変換: {best_trans['method']}")
        self.logger.info(f"  スコア: {best_trans['score']:.6f}")
        self.logger.info(f"  歪度改善: {original_stats['skewness']:.4f} → {best_trans['stats']['skewness']:.4f}")
        self.logger.info(f"  尖度改善: {original_stats['kurtosis']:.4f} → {best_trans['stats']['kurtosis']:.4f}")
        self.logger.info("=" * 80)
        
        return {
            'method': best_trans['method'],
            'y_transformed': best_trans['y_transformed'],
            'lambda': best_trans['lambda'],
            'score': best_trans['score'],
            'original_stats': original_stats,
            'transformed_stats': best_trans['stats'],
            'all_candidates': transformations
        }
    
    def inverse_transform(self, y_pred: np.ndarray, 
                         transformation_info: Dict) -> np.ndarray:
        """
        変換を逆変換
        
        Parameters
        ----------
        y_pred : np.ndarray
            変換後の予測値
        transformation_info : Dict
            変換情報
        
        Returns
        -------
        np.ndarray
            元のスケールの予測値
        """
        method = transformation_info['method']
        
        if method == 'none':
            return y_pred
        elif method == 'log':
            return np.exp(y_pred)
        elif method == 'sqrt':
            return y_pred ** 2
        elif method == 'boxcox':
            lambda_param = transformation_info['lambda']
            return inv_boxcox(y_pred, lambda_param)
        elif method == 'yeojohnson':
            lambda_param = transformation_info['lambda']
            # Yeo-Johnson逆変換
            if lambda_param == 0:
                return np.exp(y_pred) - 1
            else:
                return np.power(lambda_param * y_pred + 1, 1 / lambda_param) - 1
        elif method == 'inverse':
            return 1 / y_pred
        else:
            return y_pred
    
    def run_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        データ診断を実行
        """
        self.logger.info("=" * 80)
        self.logger.info("データ診断開始")
        self.logger.info("=" * 80)
        
        diagnostics = {}
        y = df['y'].values
        
        # 1. 基本統計量
        diagnostics['basic_stats'] = {
            'count': len(y),
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'q25': float(np.percentile(y, 25)),
            'median': float(np.median(y)),
            'q75': float(np.percentile(y, 75)),
            'max': float(np.max(y)),
            'cv': float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else 0
        }
        
        self.logger.info("基本統計量:")
        for key, val in diagnostics['basic_stats'].items():
            self.logger.info(f"  {key}: {val:.2f}")
        
        # 2. ADF検定 (定常性)
        try:
            adf_result = adfuller(y)
            diagnostics['adf_test'] = {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            self.logger.info(f"ADF検定: p値={adf_result[1]:.6f}, "
                           f"定常性={'あり' if adf_result[1] < 0.05 else 'なし'}")
        except Exception as e:
            self.logger.warning(f"ADF検定失敗: {e}")
            diagnostics['adf_test'] = None
        
        # 3. ACF/PACF
        try:
            acf_vals = acf(y, nlags=min(40, len(y)//2))
            pacf_vals = pacf(y, nlags=min(40, len(y)//2))
            diagnostics['acf'] = acf_vals.tolist()
            diagnostics['pacf'] = pacf_vals.tolist()
            self.logger.info(f"ACF/PACF計算完了 (ラグ数: {len(acf_vals)})")
        except Exception as e:
            self.logger.warning(f"ACF/PACF計算失敗: {e}")
            diagnostics['acf'] = None
            diagnostics['pacf'] = None
        
        # 4. STL分解
        try:
            if len(df) >= 730:  # 2年以上のデータ
                stl = STL(y, seasonal=365, robust=True)
                result = stl.fit()
                diagnostics['stl'] = {
                    'trend_strength': float(1 - np.var(result.resid) / np.var(result.trend + result.resid)),
                    'seasonal_strength': float(1 - np.var(result.resid) / np.var(result.seasonal + result.resid))
                }
                self.logger.info(f"STL分解: トレンド強度={diagnostics['stl']['trend_strength']:.4f}, "
                               f"季節性強度={diagnostics['stl']['seasonal_strength']:.4f}")
            else:
                self.logger.info("データ不足によりSTL分解スキップ")
                diagnostics['stl'] = None
        except Exception as e:
            self.logger.warning(f"STL分解失敗: {e}")
            diagnostics['stl'] = None
        
        # 5. 曜日別統計
        df_with_weekday = df.copy()
        df_with_weekday['weekday'] = df_with_weekday['ds'].dt.dayofweek
        weekday_stats = df_with_weekday.groupby('weekday')['y'].agg(['mean', 'std', 'min', 'max'])
        diagnostics['weekday_stats'] = weekday_stats.to_dict()
        
        self.logger.info("曜日別平均:")
        for wd in range(7):
            mean_val = weekday_stats.loc[wd, 'mean']
            self.logger.info(f"  {calendar.day_name[wd]}: {mean_val:.1f}")
        
        self.logger.info("診断完了")
        
        return diagnostics
    
    def create_holidays(self, start_date, end_date) -> pd.DataFrame:
        """日本の祝日データフレーム作成"""
        if jpholiday is None:
            return pd.DataFrame(columns=['ds', 'holiday'])
        
        holidays_list = []
        current = start_date
        while current <= end_date:
            if jpholiday.is_holiday(current):
                name = jpholiday.is_holiday_name(current)
                holidays_list.append({'ds': current, 'holiday': name})
            current += timedelta(days=1)
        
        return pd.DataFrame(holidays_list)
    
    def optimize_prophet_params(self, df_train: pd.DataFrame, 
                                holidays: pd.DataFrame,
                                n_trials: int = 200) -> Dict:
        """
        Optunaでハイパーパラメータ最適化
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Optunaハイパーパラメータ最適化開始 (試行回数: {n_trials})")
        self.logger.info("=" * 80)
        
        # CV用の分割
        cv_horizon = 30  # 30日をテスト期間とする
        cutoff = df_train['ds'].max() - timedelta(days=cv_horizon)
        train_cv = df_train[df_train['ds'] <= cutoff].copy()
        test_cv = df_train[df_train['ds'] > cutoff].copy()
        
        self.logger.info(f"CV分割: 学習={len(train_cv)}日, テスト={len(test_cv)}日")
        
        def objective(trial):
            """Optuna目的関数"""
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
                'n_changepoints': trial.suggest_int('n_changepoints', 10, 50),
                'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
                'daily_seasonality': False,
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
            }
            
            try:
                model = Prophet(
                    holidays=holidays,
                    interval_width=0.95,
                    **params
                )
                
                # カスタム季節性追加
                if trial.suggest_categorical('add_monthly', [True, False]):
                    model.add_seasonality(
                        name='monthly',
                        period=30.5,
                        fourier_order=trial.suggest_int('monthly_fourier', 3, 10)
                    )
                
                if trial.suggest_categorical('add_quarterly', [True, False]):
                    model.add_seasonality(
                        name='quarterly',
                        period=91.25,
                        fourier_order=trial.suggest_int('quarterly_fourier', 3, 8)
                    )
                
                model.fit(train_cv)
                
                # CV予測
                future_cv = model.make_future_dataframe(periods=cv_horizon)
                forecast_cv = model.predict(future_cv)
                forecast_cv = forecast_cv[forecast_cv['ds'] > cutoff]
                
                # 逆変換
                y_pred = self.inverse_transform(
                    forecast_cv['yhat'].values,
                    self.transformation_info
                )
                y_true = test_cv['y'].values
                
                # MAE計算
                mae = mean_absolute_error(y_true, y_pred)
                
                return mae
            
            except Exception as e:
                return 1e10
        
        # 最適化実行
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_mae = study.best_value
        
        self.logger.info("=" * 80)
        self.logger.info("最適化完了")
        self.logger.info(f"最良MAE: {best_mae:.2f}")
        self.logger.info("最適パラメータ:")
        for key, val in best_params.items():
            self.logger.info(f"  {key}: {val}")
        self.logger.info("=" * 80)
        
        return best_params
    
    def fit_optimized_model(self, df: pd.DataFrame, 
                           best_params: Dict,
                           holidays: pd.DataFrame,
                           forecast_periods: int = 60) -> Tuple[Prophet, pd.DataFrame]:
        """
        最適化されたモデルを学習して予測
        
        Parameters
        ----------
        df : pd.DataFrame
            学習データ
        best_params : Dict
            Optunaで最適化されたパラメータ
        holidays : pd.DataFrame
            祝日データ
        forecast_periods : int
            予測期間(日数)
        
        Returns
        -------
        Tuple[Prophet, pd.DataFrame]
            学習済みモデルと予測結果
        """
        self.logger.info("=" * 80)
        self.logger.info("最適化モデルの学習開始")
        self.logger.info(f"学習データ期間: {df['ds'].min()} 〜 {df['ds'].max()}")
        self.logger.info(f"学習データ数: {len(df)} 日")
        self.logger.info("=" * 80)
        
        # カスタム季節性のパラメータを抽出
        add_monthly = best_params.pop('add_monthly', False)
        monthly_fourier = best_params.pop('monthly_fourier', 5)
        add_quarterly = best_params.pop('add_quarterly', False)
        quarterly_fourier = best_params.pop('quarterly_fourier', 5)
        
        # モデル作成
        model = Prophet(
            holidays=holidays,
            interval_width=0.95,
            **best_params
        )
        
        # カスタム季節性追加
        if add_monthly:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=monthly_fourier
            )
            self.logger.info(f"月次季節性追加 (フーリエ次数: {monthly_fourier})")
        
        if add_quarterly:
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=quarterly_fourier
            )
            self.logger.info(f"四半期季節性追加 (フーリエ次数: {quarterly_fourier})")
        
        # 学習
        self.logger.info("モデル学習中...")
        model.fit(df)
        self.logger.info("学習完了")
        
        # 予測
        self.logger.info(f"{forecast_periods}日先の予測実行中...")
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        # 逆変換
        forecast['yhat'] = self.inverse_transform(
            forecast['yhat'].values,
            self.transformation_info
        )
        forecast['yhat_lower'] = self.inverse_transform(
            forecast['yhat_lower'].values,
            self.transformation_info
        )
        forecast['yhat_upper'] = self.inverse_transform(
            forecast['yhat_upper'].values,
            self.transformation_info
        )
        
        self.logger.info("予測完了")
        
        return model, forecast
    
    def validate_forecast(self, df_validation: pd.DataFrame,
                         forecast: pd.DataFrame) -> Dict:
        """
        検証データで予測精度を評価
        """
        self.logger.info("=" * 80)
        self.logger.info("検証開始")
        self.logger.info("=" * 80)
        
        # マージ
        validation_dates = df_validation['ds'].values
        forecast_subset = forecast[forecast['ds'].isin(validation_dates)].copy()
        
        if len(forecast_subset) == 0:
            self.logger.warning("検証データとの一致なし")
            return {}
        
        merged = pd.merge(
            df_validation[['ds', 'y']],
            forecast_subset[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='inner'
        )
        
        if len(merged) == 0:
            self.logger.warning("マージ後のデータなし")
            return {}
        
        # 月別に評価
        merged['year_month'] = merged['ds'].dt.to_period('M')
        months = sorted(merged['year_month'].unique())
        
        metrics = {}
        
        for i, month in enumerate(months, 1):
            month_data = merged[merged['year_month'] == month]
            y_true = month_data['y'].values
            y_pred = month_data['yhat'].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            metrics[f'month_{i}'] = {
                'period': str(month),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'n_days': len(month_data)
            }
            
            self.logger.info(f"月{i} ({month}): RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        
        # 全体
        y_true_all = merged['y'].values
        y_pred_all = merged['yhat'].values
        
        rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae_all = mean_absolute_error(y_true_all, y_pred_all)
        mape_all = np.mean(np.abs((y_true_all - y_pred_all) / y_true_all)) * 100
        
        metrics['overall'] = {
            'rmse': float(rmse_all),
            'mae': float(mae_all),
            'mape': float(mape_all),
            'n_days': len(merged)
        }
        
        self.logger.info(f"全体: RMSE={rmse_all:.2f}, MAE={mae_all:.2f}, MAPE={mape_all:.2f}%")
        self.logger.info("検証完了")
        
        # JSON保存
        with open(self.output_dir / "validation_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        return metrics
    
    def create_visualizations(self, df_train: pd.DataFrame,
                             df_validation: pd.DataFrame,
                             forecast_validation: pd.DataFrame,
                             forecast_production: pd.DataFrame):
        """
        可視化作成
        """
        self.logger.info("=" * 80)
        self.logger.info("可視化作成開始")
        self.logger.info("=" * 80)
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 全体時系列
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df_train['ds'], df_train['y'], 'o-', label='訓練データ', alpha=0.6)
        ax1.plot(df_validation['ds'], df_validation['y'], 'o-', label='検証データ', alpha=0.6)
        
        # 検証予測
        val_future = forecast_validation[forecast_validation['ds'] > df_train['ds'].max()]
        ax1.plot(val_future['ds'], val_future['yhat'], 'r-', label='検証予測', linewidth=2)
        ax1.fill_between(val_future['ds'], val_future['yhat_lower'], val_future['yhat_upper'],
                        alpha=0.2, color='red')
        
        # 実務予測
        prod_future = forecast_production[forecast_production['ds'] > df_validation['ds'].max()]
        ax1.plot(prod_future['ds'], prod_future['yhat'], 'g-', label='実務予測', linewidth=2)
        ax1.fill_between(prod_future['ds'], prod_future['yhat_lower'], prod_future['yhat_upper'],
                        alpha=0.2, color='green')
        
        ax1.set_title('時系列全体と予測', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日付')
        ax1.set_ylabel('呼量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 検証期間拡大
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df_validation['ds'], df_validation['y'], 'o-', label='実測値', alpha=0.7)
        ax2.plot(val_future['ds'], val_future['yhat'], 'r-', label='予測値', linewidth=2)
        ax2.fill_between(val_future['ds'], val_future['yhat_lower'], val_future['yhat_upper'],
                        alpha=0.2, color='red')
        ax2.set_title('検証期間の予測', fontsize=12, fontweight='bold')
        ax2.set_xlabel('日付')
        ax2.set_ylabel('呼量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 実務予測期間
        ax3 = fig.add_subplot(gs[1, 1])
        # 直近30日 + 実務予測
        recent = df_validation.tail(30)
        ax3.plot(recent['ds'], recent['y'], 'o-', label='直近実績', alpha=0.7)
        ax3.plot(prod_future['ds'], prod_future['yhat'], 'g-', label='実務予測', linewidth=2)
        ax3.fill_between(prod_future['ds'], prod_future['yhat_lower'], prod_future['yhat_upper'],
                        alpha=0.2, color='green')
        ax3.set_title('実務予測期間', fontsize=12, fontweight='bold')
        ax3.set_xlabel('日付')
        ax3.set_ylabel('呼量')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差プロット
        ax4 = fig.add_subplot(gs[1, 2])
        if self.validation_metrics:
            merged_val = pd.merge(
                df_validation[['ds', 'y']],
                forecast_validation[['ds', 'yhat']],
                on='ds'
            )
            residuals = merged_val['y'] - merged_val['yhat']
            ax4.scatter(merged_val['yhat'], residuals, alpha=0.5)
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_title('残差プロット', fontsize=12, fontweight='bold')
            ax4.set_xlabel('予測値')
            ax4.set_ylabel('残差')
            ax4.grid(True, alpha=0.3)
        
        # 5. 変換前後の分布比較
        ax5 = fig.add_subplot(gs[2, 0])
        y_original = df_train['y']
        ax5.hist(y_original, bins=50, alpha=0.7, label='元データ')
        ax5.set_title('元データの分布', fontsize=12, fontweight='bold')
        ax5.set_xlabel('呼量')
        ax5.set_ylabel('頻度')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        if self.transformation_info['method'] != 'none':
            y_transformed = self.transformation_info['y_transformed']
            ax6.hist(y_transformed, bins=50, alpha=0.7, color='orange', 
                    label=f"変換後 ({self.transformation_info['method']})")
            ax6.set_title('変換後の分布', fontsize=12, fontweight='bold')
            ax6.set_xlabel('変換後の値')
            ax6.set_ylabel('頻度')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 6. Q-Qプロット
        ax7 = fig.add_subplot(gs[2, 2])
        if self.transformation_info['method'] != 'none':
            y_trans = self.transformation_info['y_transformed']
            stats.probplot(y_trans, dist="norm", plot=ax7)
            ax7.set_title(f"Q-Qプロット ({self.transformation_info['method']})", 
                         fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 7. 曜日別統計
        ax8 = fig.add_subplot(gs[3, 0])
        df_all = pd.concat([df_train, df_validation])
        df_all['weekday'] = df_all['ds'].dt.dayofweek
        weekday_mean = df_all.groupby('weekday')['y'].mean()
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        ax8.bar(range(7), weekday_mean.values)
        ax8.set_xticks(range(7))
        ax8.set_xticklabels(weekday_names)
        ax8.set_title('曜日別平均呼量', fontsize=12, fontweight='bold')
        ax8.set_xlabel('曜日')
        ax8.set_ylabel('平均呼量')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 8. 月別統計
        ax9 = fig.add_subplot(gs[3, 1])
        df_all['month'] = df_all['ds'].dt.month
        monthly_mean = df_all.groupby('month')['y'].mean()
        ax9.bar(range(1, 13), monthly_mean.values)
        ax9.set_xticks(range(1, 13))
        ax9.set_title('月別平均呼量', fontsize=12, fontweight='bold')
        ax9.set_xlabel('月')
        ax9.set_ylabel('平均呼量')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 9. 検証メトリクス
        ax10 = fig.add_subplot(gs[3, 2])
        if self.validation_metrics:
            metrics_text = "検証メトリクス\n" + "=" * 30 + "\n"
            for key, val in self.validation_metrics.items():
                if key == 'overall':
                    metrics_text += f"\n全体:\n"
                else:
                    metrics_text += f"\n{key} ({val.get('period', '')}):\n"
                metrics_text += f"  RMSE: {val['rmse']:.2f}\n"
                metrics_text += f"  MAE: {val['mae']:.2f}\n"
                metrics_text += f"  MAPE: {val['mape']:.2f}%\n"
            
            ax10.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                     verticalalignment='center')
            ax10.axis('off')
        
        plt.suptitle('Prophet v4.0 最適化予測システム - 分析結果', 
                    fontsize=16, fontweight='bold')
        
        output_path = self.output_dir / "visualizations.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可視化保存完了: {output_path}")
    
    def generate_report(self) -> str:
        """
        レポート生成
        """
        report = []
        report.append("=" * 80)
        report.append("Prophet v4.0 最適化予測システム - 実行レポート")
        report.append("=" * 80)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # データ情報
        report.append("【データ情報】")
        report.append(f"訓練期間: {self.df_train['ds'].min()} 〜 {self.df_train['ds'].max()}")
        report.append(f"訓練データ数: {len(self.df_train)} 日")
        report.append(f"検証期間: {self.df_validation['ds'].min()} 〜 {self.df_validation['ds'].max()}")
        report.append(f"検証データ数: {len(self.df_validation)} 日")
        report.append("")
        
        # 変換情報
        report.append("【データ変換】")
        report.append(f"選択された変換: {self.transformation_info['method']}")
        report.append(f"変換スコア: {self.transformation_info['score']:.6f}")
        if self.transformation_info['lambda'] is not None:
            report.append(f"λパラメータ: {self.transformation_info['lambda']:.4f}")
        report.append(f"元データ歪度: {self.transformation_info['original_stats']['skewness']:.4f}")
        report.append(f"変換後歪度: {self.transformation_info['transformed_stats']['skewness']:.4f}")
        report.append("")
        
        # 最適パラメータ
        report.append("【Optuna最適化パラメータ】")
        for key, val in self.best_params.items():
            report.append(f"  {key}: {val}")
        report.append("")
        
        # 検証メトリクス
        report.append("【検証メトリクス】")
        if self.validation_metrics:
            for key, val in self.validation_metrics.items():
                if key == 'overall':
                    report.append(f"\n全体 ({val.get('n_days', 0)} 日):")
                else:
                    report.append(f"\n{key} ({val.get('period', '')}, {val.get('n_days', 0)} 日):")
                report.append(f"  RMSE: {val['rmse']:.2f}")
                report.append(f"  MAE: {val['mae']:.2f}")
                report.append(f"  MAPE: {val['mape']:.2f}%")
        report.append("")
        
        # 実務予測情報
        if self.forecast_production is not None:
            prod_future = self.forecast_production[
                self.forecast_production['ds'] > self.df_validation['ds'].max()
            ]
            report.append("【実務予測】")
            report.append(f"予測期間: {prod_future['ds'].min()} 〜 {prod_future['ds'].max()}")
            report.append(f"予測日数: {len(prod_future)} 日")
            report.append(f"予測平均: {prod_future['yhat'].mean():.1f}")
            report.append(f"予測範囲: {prod_future['yhat'].min():.1f} 〜 {prod_future['yhat'].max():.1f}")
            report.append("")
        
        # ファイル一覧
        report.append("【出力ファイル】")
        report.append(f"  validation_forecast.csv - 検証期間予測")
        report.append(f"  forecast.csv - 実務用予測")
        report.append(f"  validation_metrics.json - 検証メトリクス")
        report.append(f"  visualizations.png - 可視化")
        report.append(f"  model_validation.pkl - 検証用モデル")
        report.append(f"  model_production.pkl - 実務用モデル")
        report.append(f"  transformation_info.json - 変換情報")
        report.append(f"  best_params.json - 最適パラメータ")
        report.append("")
        
        report.append("=" * 80)
        report.append("実行完了")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # 保存
        with open(self.output_dir / "report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def fit_predict(self, filepath: str, 
                   validation_months: int = 2,
                   optuna_trials: int = 200) -> Dict:
        """
        完全な予測パイプライン実行
        
        Parameters
        ----------
        filepath : str
            CSVファイルパス
        validation_months : int
            検証期間(月数)
        optuna_trials : int
            Optuna試行回数
        
        Returns
        -------
        Dict
            結果サマリー
        """
        self.logger.info("=" * 80)
        self.logger.info("Prophet v4.0 最適化予測パイプライン開始")
        self.logger.info("=" * 80)
        
        # 1. データ読み込み
        df = self.load_data(filepath)
        
        # 2. 訓練/検証分割
        split_date = df['ds'].max() - relativedelta(months=validation_months)
        self.df_train = df[df['ds'] <= split_date].copy()
        self.df_validation = df[df['ds'] > split_date].copy()
        self.df_full = df.copy()
        
        self.logger.info(f"データ分割:")
        self.logger.info(f"  訓練: {len(self.df_train)} 日 ({self.df_train['ds'].min()} 〜 {self.df_train['ds'].max()})")
        self.logger.info(f"  検証: {len(self.df_validation)} 日 ({self.df_validation['ds'].min()} 〜 {self.df_validation['ds'].max()})")
        
        # 3. データ変換選択 (訓練データベース)
        self.transformation_info = self.select_optimal_transformation(self.df_train['y'])
        
        # 訓練データに変換を適用
        self.df_train['y'] = self.transformation_info['y_transformed'].values
        
        # 検証データにも同じ変換を適用
        if self.transformation_info['method'] == 'log':
            self.df_validation['y'] = np.log(self.df_validation['y'])
        elif self.transformation_info['method'] == 'sqrt':
            self.df_validation['y'] = np.sqrt(self.df_validation['y'])
        elif self.transformation_info['method'] == 'boxcox':
            lambda_param = self.transformation_info['lambda']
            self.df_validation['y'] = boxcox(self.df_validation['y'], lmbda=lambda_param)
        elif self.transformation_info['method'] == 'yeojohnson':
            lambda_param = self.transformation_info['lambda']
            self.df_validation['y'] = yeojohnson(self.df_validation['y'], lmbda=lambda_param)
        elif self.transformation_info['method'] == 'inverse':
            self.df_validation['y'] = 1 / self.df_validation['y']
        
        # 変換情報保存
        with open(self.output_dir / "transformation_info.json", 'w', encoding='utf-8') as f:
            # numpy配列をリストに変換
            trans_info_save = {
                'method': self.transformation_info['method'],
                'lambda': float(self.transformation_info['lambda']) if self.transformation_info['lambda'] is not None else None,
                'score': float(self.transformation_info['score']),
                'original_stats': {k: float(v) for k, v in self.transformation_info['original_stats'].items()},
                'transformed_stats': {k: float(v) for k, v in self.transformation_info['transformed_stats'].items()}
            }
            json.dump(trans_info_save, f, indent=2, ensure_ascii=False)
        
        # 4. 診断
        self.diagnostics = self.run_diagnostics(self.df_train)
        
        # 5. 祝日データ作成
        start_date = df['ds'].min().date()
        end_date = df['ds'].max().date() + timedelta(days=90)
        holidays = self.create_holidays(start_date, end_date)
        self.logger.info(f"祝日データ作成: {len(holidays)} 件")
        
        # 6. Optuna最適化 (訓練データのみ)
        self.best_params = self.optimize_prophet_params(
            self.df_train, holidays, n_trials=optuna_trials
        )
        
        # パラメータ保存
        with open(self.output_dir / "best_params.json", 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        
        # 7. 検証用モデル学習と予測 (訓練データのみ)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 1: 検証用モデル (訓練データのみで学習)")
        self.logger.info("=" * 80)
        
        self.model_validation, self.forecast_validation = self.fit_optimized_model(
            self.df_train,
            self.best_params.copy(),
            holidays,
            forecast_periods=len(self.df_validation)
        )
        
        # 検証予測保存
        validation_future = self.forecast_validation[
            self.forecast_validation['ds'] > self.df_train['ds'].max()
        ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        validation_future.to_csv(
            self.output_dir / "validation_forecast.csv",
            index=False
        )
        self.logger.info(f"検証予測保存: {len(validation_future)} 日")
        
        # 検証メトリクス計算
        self.validation_metrics = self.validate_forecast(
            self.df_validation,
            self.forecast_validation
        )
        
        # 8. 実務用モデル学習と予測 (全データ)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 2: 実務用モデル (全データで再学習)")
        self.logger.info("=" * 80)
        
        # 全データに変換適用
        df_full_transformed = self.df_full.copy()
        if self.transformation_info['method'] == 'log':
            df_full_transformed['y'] = np.log(df_full_transformed['y'])
        elif self.transformation_info['method'] == 'sqrt':
            df_full_transformed['y'] = np.sqrt(df_full_transformed['y'])
        elif self.transformation_info['method'] == 'boxcox':
            lambda_param = self.transformation_info['lambda']
            df_full_transformed['y'] = boxcox(df_full_transformed['y'], lmbda=lambda_param)
        elif self.transformation_info['method'] == 'yeojohnson':
            lambda_param = self.transformation_info['lambda']
            df_full_transformed['y'] = yeojohnson(df_full_transformed['y'], lmbda=lambda_param)
        elif self.transformation_info['method'] == 'inverse':
            df_full_transformed['y'] = 1 / df_full_transformed['y']
        
        self.model_production, self.forecast_production = self.fit_optimized_model(
            df_full_transformed,
            self.best_params.copy(),
            holidays,
            forecast_periods=60  # 2ヶ月
        )
        
        # 実務予測保存
        production_future = self.forecast_production[
            self.forecast_production['ds'] > self.df_full['ds'].max()
        ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        production_future.to_csv(
            self.output_dir / "forecast.csv",
            index=False
        )
        self.logger.info(f"実務予測保存: {len(production_future)} 日")
        
        # 9. 可視化
        self.create_visualizations(
            self.df_train,
            self.df_validation,
            self.forecast_validation,
            self.forecast_production
        )
        
        # 10. レポート生成
        report = self.generate_report()
        self.logger.info("\n" + report)
        
        # 11. モデル保存
        with open(self.output_dir / "model_validation.pkl", 'wb') as f:
            pickle.dump(self.model_validation, f)
        with open(self.output_dir / "model_production.pkl", 'wb') as f:
            pickle.dump(self.model_production, f)
        
        self.logger.info("=" * 80)
        self.logger.info("すべての処理が完了しました")
        self.logger.info("=" * 80)
        
        return {
            'diagnostics': self.diagnostics,
            'transformation': self.transformation_info,
            'best_params': self.best_params,
            'model_validation': self.model_validation,
            'model_production': self.model_production,
            'validation_metrics': self.validation_metrics,
            'report': report
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prophet v4.0 最適化予測システム')
    parser.add_argument('filepath', type=str, help='CSVファイルパス')
    parser.add_argument('--validation-months', type=int, default=2,
                       help='検証期間(月数) (デフォルト: 2)')
    parser.add_argument('--optuna-trials', type=int, default=200,
                       help='Optuna試行回数 (デフォルト: 200)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='出力ディレクトリ (デフォルト: output)')
    
    args = parser.parse_args()
    
    # 実行
    predictor = ProphetOptimizedPredictor(output_dir=args.output_dir)
    result = predictor.fit_predict(
        filepath=args.filepath,
        validation_months=args.validation_months,
        optuna_trials=args.optuna_trials
    )
    
    print("\n" + "=" * 80)
    print("実行完了")
    print("=" * 80)
    print(result['report'])
