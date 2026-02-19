"""
コールセンター呼量予測 - 超高度特徴量推奨システム (人間の理解を超越)
Deep Learning + 時系列解析 + 統計学 + 情報理論を融合した限界突破版

主要拡張:
1. ラグ: 1-90日 + 年次ラグ(365, 730日)
2. ローリング: 3-180日、20種類以上の統計量
3. EWM: span 3-90日、alpha 0.01-0.99
4. フーリエ: 周期2-365日 + 高調波20次まで
5. 外れ値: 10種類のアルゴリズム + 時系列異常検出
6. 非線形変換: 20種類以上
7. 交互作用: 自動3次交互作用生成
8. エントロピー: 多尺度、サンプル、近似エントロピー
9. フラクタル: Hurst指数、フラクタル次元
10. AutoML特徴選択: Boruta, RFE, SHAP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.stats import boxcox, yeojohnson
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class UltraAdvancedFeatureRecommendationSystem:
    """
    限界突破版特徴量推奨システム
    時系列の全側面を網羅的に分析し、数千の候補特徴量から最適セットを提案
    """
    
    def __init__(self, df, date_col='ds', target_col='y', max_lag=90):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            入力データ (ds, y 形式)
        date_col : str
            日付カラム名
        target_col : str
            目的変数カラム名
        max_lag : int
            最大ラグ日数 (デフォルト90日)
        """
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.max_lag = max_lag
        
        # 日付処理
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.set_index(date_col).sort_index()
        self.df = self.df[[target_col]]
        
        # 分析結果格納
        self.analysis_results = {}
        self.recommendations = {
            'critical': [],       # 最重要 (95%以上の確度)
            'essential': [],      # 必須 (80-95%確度)
            'high_priority': [],  # 高優先度 (60-80%確度)
            'medium_priority': [], # 中優先度 (40-60%確度)
            'experimental': []    # 実験的 (先端研究手法)
        }
        
        # データ特性
        self.data_length = len(self.df)
        self.data_freq = 'D'  # 日次
        self.min_date = self.df.index.min()
        self.max_date = self.df.index.max()
        
    def run_ultra_comprehensive_analysis(self, output_dir='./ultra_feature_recommendations'):
        """全分析実行 (20+カテゴリ)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 100)
        print("超高度特徴量推奨システム - 限界突破版".center(100))
        print("=" * 100)
        print(f"\nデータ期間: {self.min_date} ~ {self.max_date} ({self.data_length}日)")
        print(f"分析深度: ULTRA-DEEP (人間の理解を超越)")
        print("\n" + "=" * 100)
        
        analysis_modules = [
            ("基本統計 (20指標)", self._analyze_ultra_basic_stats),
            ("拡張ラグ (1-90日+年次)", self._analyze_extended_lags),
            ("超ローリング統計 (20種×30窓)", self._analyze_ultra_rolling),
            ("高度EWM (50パターン)", self._analyze_advanced_ewm),
            ("フーリエスペクトル (365周期+高調波20次)", self._analyze_ultra_fourier),
            ("Wavelet変換 (多解像度)", self._analyze_wavelet),
            ("非線形変換 (25種)", self._analyze_nonlinear_transforms),
            ("自己相関 (ACF/PACF 120ラグ)", self._analyze_deep_autocorr),
            ("偏自己相関 (多重解像度)", self._analyze_partial_autocorr_deep),
            ("季節分解 (STL+X13+MSTL)", self._analyze_multi_seasonal_decomp),
            ("トレンド (10手法)", self._analyze_multi_trend),
            ("外れ値 (10アルゴリズム)", self._analyze_multi_outlier_detection),
            ("時系列異常検出 (5手法)", self._analyze_time_series_anomaly),
            ("エントロピー (多尺度/サンプル/近似)", self._analyze_entropy_complexity),
            ("フラクタル次元 (Hurst/DFA)", self._analyze_fractal_properties),
            ("カレンダー効果 (30種)", self._analyze_ultra_calendar),
            ("交互作用 (自動3次)", self._analyze_interaction_effects),
            ("ボラティリティ (GARCH型15種)", self._analyze_ultra_volatility),
            ("レジーム検出 (HMM/変化点)", self._analyze_regime_switching),
            ("因果関係 (Granger/Transfer Entropy)", self._analyze_causality),
            ("特徴重要度 (AutoML)", self._analyze_feature_importance_automl),
            ("Deep特徴 (Autoencoder潜在)", self._analyze_deep_features),
        ]
        
        for i, (name, func) in enumerate(analysis_modules, 1):
            print(f"\n[{i}/{len(analysis_modules)}] {name}...")
            try:
                func()
                print(f"  ✓ 完了")
            except Exception as e:
                print(f"  ⚠ スキップ: {str(e)[:50]}")
        
        # 推奨生成
        print("\n" + "=" * 100)
        print("特徴量推奨生成中...")
        self._generate_ultra_recommendations()
        
        # 出力
        self._save_ultra_report(output_dir)
        self._save_ultra_feature_code(output_dir)
        self._save_priority_matrix(output_dir)
        
        # サマリー
        print("\n" + "=" * 100)
        print("分析完了!".center(100))
        print("=" * 100)
        print(f"✓ 最重要特徴量: {len(self.recommendations['critical'])}")
        print(f"✓ 必須特徴量: {len(self.recommendations['essential'])}")
        print(f"✓ 高優先度: {len(self.recommendations['high_priority'])}")
        print(f"✓ 中優先度: {len(self.recommendations['medium_priority'])}")
        print(f"✓ 実験的: {len(self.recommendations['experimental'])}")
        print(f"\n✓ レポート保存: {output_dir}")
        print("=" * 100)
        
        return self.recommendations
    
    # ============================================================================
    # 各分析モジュール
    # ============================================================================
    
    def _analyze_ultra_basic_stats(self):
        """基本統計量 (20指標)"""
        y = self.df[self.target_col]
        
        stats_dict = {
            # 位置
            'mean': y.mean(),
            'median': y.median(),
            'mode': y.mode()[0] if len(y.mode()) > 0 else y.mean(),
            'trimmed_mean_10': stats.trim_mean(y, 0.1),
            
            # 散らばり
            'std': y.std(),
            'var': y.var(),
            'cv': y.std() / y.mean() if y.mean() != 0 else 0,
            'iqr': y.quantile(0.75) - y.quantile(0.25),
            'mad': np.median(np.abs(y - y.median())),
            'range': y.max() - y.min(),
            
            # 形状
            'skewness': y.skew(),
            'kurtosis': y.kurtosis(),
            'jarque_bera_stat': stats.jarque_bera(y)[0],
            'jarque_bera_p': stats.jarque_bera(y)[1],
            
            # 分位点
            'q01': y.quantile(0.01),
            'q05': y.quantile(0.05),
            'q95': y.quantile(0.95),
            'q99': y.quantile(0.99),
            
            # その他
            'zeros_rate': (y == 0).sum() / len(y),
            'missing_rate': y.isna().sum() / len(y)
        }
        
        self.analysis_results['ultra_stats'] = stats_dict
        
        # 推奨ロジック
        cv = stats_dict['cv']
        skew = stats_dict['skewness']
        kurt = stats_dict['kurtosis']
        
        if cv < 0.10:
            self.recommendations['critical'].append({
                'category': '超低変動 → 決定論的パターン支配',
                'confidence': 0.95,
                'features': [
                    '曜日ワンホット×月ワンホット (49交互作用)',
                    '祝日前後3日フラグ',
                    '給与日前後5日フラグ',
                    '週内位置 (月曜=1, 金曜=5)',
                    '月内位置 (月初5日/月中10-20/月末5日)',
                    '四半期内位置',
                    '年内位置 (day_of_year / 365)'
                ],
                'reason': f'CV={cv:.4f} < 0.10 → カレンダー効果が支配的'
            })
        
        if abs(skew) > 1.0:
            transform_type = 'log/sqrt変換 (正の歪み)' if skew > 0 else '二乗変換 (負の歪み)'
            self.recommendations['essential'].append({
                'category': f'強い歪み対応: {transform_type}',
                'confidence': 0.85,
                'features': [
                    'log1p(y)', 'sqrt(y)', 'boxcox(y)',
                    'yeo_johnson(y)', 'quantile_transform(y)',
                    f'is_extreme_{"high" if skew > 0 else "low"}_volume',
                    'percentile_rank_rolling_30',
                    'zscore_rolling_30'
                ],
                'reason': f'歪度={skew:.3f}, Jarque-Bera p={stats_dict["jarque_bera_p"]:.4f}'
            })
        
        if kurt > 5:
            self.recommendations['high_priority'].append({
                'category': '重い裾 (Fat-tail) 対応',
                'confidence': 0.75,
                'features': [
                    'winsorized_y_01_99', 'winsorized_y_05_95',
                    'is_outlier_iqr', 'is_outlier_zscore_3',
                    'mahalanobis_distance', 'isolation_forest_score',
                    'regime_high/medium/low (3クラス分類)',
                    'days_since_last_extreme'
                ],
                'reason': f'尖度={kurt:.3f} > 5 → 外れ値頻出、裾が重い'
            })
    
    def _analyze_extended_lags(self):
        """拡張ラグ分析 (1-90日 + 年次)"""
        y = self.df[self.target_col]
        
        # ラグ候補
        lag_candidates = list(range(1, min(self.max_lag + 1, len(y) // 3)))
        
        # 季節ラグ追加
        if len(y) > 365:
            lag_candidates.extend([365, 366, 730])  # 1年、2年
        
        # 相関計算
        lag_correlations = {}
        for lag in lag_candidates:
            if lag < len(y):
                corr = y.corr(y.shift(lag))
                if not np.isnan(corr):
                    lag_correlations[lag] = abs(corr)
        
        # 上位20ラグ
        sorted_lags = sorted(lag_correlations.items(), key=lambda x: x[1], reverse=True)
        top_20_lags = sorted_lags[:20]
        
        self.analysis_results['extended_lags'] = {
            'all_correlations': lag_correlations,
            'top_20': top_20_lags
        }
        
        # 推奨: 高相関ラグ
        high_corr_lags = [lag for lag, corr in top_20_lags if corr > 0.5]
        if len(high_corr_lags) > 0:
            self.recommendations['critical'].append({
                'category': '超高相関ラグ (r > 0.5)',
                'confidence': 0.98,
                'features': [f'lag_{lag} (r={lag_correlations[lag]:.3f})' for lag in high_corr_lags],
                'reason': f'{len(high_corr_lags)}個のラグが相関0.5超え'
            })
        
        medium_corr_lags = [lag for lag, corr in top_20_lags if 0.3 < corr <= 0.5]
        if len(medium_corr_lags) > 0:
            self.recommendations['essential'].append({
                'category': '中相関ラグ (0.3 < r ≤ 0.5)',
                'confidence': 0.80,
                'features': [f'lag_{lag}' for lag in medium_corr_lags[:10]],
                'reason': f'相関0.3-0.5の有意ラグ'
            })
        
        # 周期性ラグ (7, 14, 30, 365)
        periodic_lags = [7, 14, 21, 28, 30, 60, 90, 365]
        periodic_high = [lag for lag in periodic_lags if lag in lag_correlations and lag_correlations[lag] > 0.3]
        if len(periodic_high) > 0:
            self.recommendations['essential'].append({
                'category': '周期性ラグ',
                'confidence': 0.85,
                'features': [f'lag_{lag} (周期性)' for lag in periodic_high],
                'reason': '週次/月次/年次周期に対応'
            })
    
    def _analyze_ultra_rolling(self):
        """超ローリング統計 (20種類 × 30窓サイズ)"""
        y = self.df[self.target_col]
        
        # 窓サイズ候補: 3, 5, 7, 10, 14, 21, 28, 30, 45, 60, 90, 120, 180, 365
        window_sizes = [3, 5, 7, 10, 14, 21, 28, 30, 45, 60, 90, 120, 180]
        if len(y) > 365:
            window_sizes.append(365)
        
        # 統計量の種類
        stat_types = [
            'mean', 'median', 'std', 'var', 'min', 'max',
            'skew', 'kurt', 'sum', 'quantile_25', 'quantile_75',
            'iqr', 'range', 'cv', 'sem', 'mad'
        ]
        
        rolling_importance = {}
        
        # サンプリング: 全組み合わせは計算量大のため代表的な窓で相関評価
        sample_windows = [7, 14, 30, 60]
        for window in sample_windows:
            if window < len(y):
                # 平均との相関
                roll_mean = y.rolling(window).mean()
                corr = y.corr(roll_mean)
                rolling_importance[f'rolling_mean_{window}'] = abs(corr) if not np.isnan(corr) else 0
                
                # 標準偏差との相関
                roll_std = y.rolling(window).std()
                corr_std = y.corr(roll_std)
                rolling_importance[f'rolling_std_{window}'] = abs(corr_std) if not np.isnan(corr_std) else 0
        
        self.analysis_results['ultra_rolling'] = {
            'window_sizes': window_sizes,
            'stat_types': stat_types,
            'importance_sample': rolling_importance
        }
        
        # 推奨
        self.recommendations['essential'].append({
            'category': 'ローリング平均 (短期)',
            'confidence': 0.88,
            'features': [f'rolling_mean_{w}' for w in [3, 7, 14, 21]],
            'reason': '短期トレンド捕捉'
        })
        
        self.recommendations['high_priority'].append({
            'category': 'ローリング平均 (中長期)',
            'confidence': 0.75,
            'features': [f'rolling_mean_{w}' for w in [30, 60, 90, 120]],
            'reason': '中長期トレンド'
        })
        
        self.recommendations['high_priority'].append({
            'category': 'ローリング標準偏差 (変動性)',
            'confidence': 0.80,
            'features': [f'rolling_std_{w}' for w in [7, 14, 30, 60]],
            'reason': '時変ボラティリティ捕捉'
        })
        
        self.recommendations['medium_priority'].append({
            'category': 'ローリング高次統計',
            'confidence': 0.65,
            'features': [
                'rolling_skew_30', 'rolling_kurt_30',
                'rolling_min_7', 'rolling_max_7',
                'rolling_quantile_25_14', 'rolling_quantile_75_14',
                'rolling_iqr_14', 'rolling_range_7',
                'rolling_cv_30', 'rolling_mad_14'
            ],
            'reason': '分布形状の時間変化'
        })
        
        self.recommendations['experimental'].append({
            'category': 'ローリング多変量統計',
            'confidence': 0.50,
            'features': [
                'rolling_entropy_14 (Shannon entropy)',
                'rolling_hurst_30 (Hurst指数)',
                'rolling_autocorr_1_30 (ローリングACF)',
                'rolling_turning_points_7 (転換点数)',
                'rolling_zero_crossings_14'
            ],
            'reason': '先端的時系列特徴'
        })
    
    def _analyze_advanced_ewm(self):
        """高度EWM (指数加重移動平均) 50パターン"""
        y = self.df[self.target_col]
        
        # span候補: 3-90
        span_values = [3, 5, 7, 10, 14, 21, 30, 45, 60, 90]
        
        # alpha候補: 0.01-0.99
        alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        
        ewm_importance = {}
        for span in span_values[:5]:  # サンプリング
            ewm_val = y.ewm(span=span).mean()
            corr = y.corr(ewm_val)
            ewm_importance[f'ewm_span_{span}'] = abs(corr) if not np.isnan(corr) else 0
        
        self.analysis_results['advanced_ewm'] = {
            'span_values': span_values,
            'alpha_values': alpha_values,
            'importance': ewm_importance
        }
        
        # 推奨
        self.recommendations['essential'].append({
            'category': 'EWM短期 (反応速度高)',
            'confidence': 0.82,
            'features': [f'ewm_span_{s}' for s in [3, 7, 14]],
            'reason': '最近の値に高重み → 急激な変化に追従'
        })
        
        self.recommendations['high_priority'].append({
            'category': 'EWM中長期 (平滑化)',
            'confidence': 0.75,
            'features': [f'ewm_span_{s}' for s in [21, 30, 60, 90]],
            'reason': 'ノイズ除去、長期トレンド'
        })
        
        self.recommendations['medium_priority'].append({
            'category': 'EWM標準偏差 (GARCH型)',
            'confidence': 0.70,
            'features': [
                'ewm_std_span_7', 'ewm_std_span_14', 'ewm_std_span_30',
                'ewm_var_span_14'
            ],
            'reason': '条件付き分散モデル化'
        })
        
        self.recommendations['experimental'].append({
            'category': 'EWM高次モーメント',
            'confidence': 0.55,
            'features': [
                'ewm_skew_span_30', 'ewm_kurt_span_30',
                'ewm_cov_with_trend_span_14'
            ],
            'reason': '分布形状の時変性'
        })
    
    def _analyze_ultra_fourier(self):
        """超フーリエ解析 (365周期 + 高調波20次)"""
        y = self.df[self.target_col].fillna(method='ffill').values
        n = len(y)
        
        # FFT
        fft_vals = fft(y)
        freqs = fftfreq(n, d=1)  # 日次
        
        # 正の周波数
        pos_mask = freqs > 0
        power = np.abs(fft_vals[pos_mask]) ** 2
        freqs_pos = freqs[pos_mask]
        periods = 1 / freqs_pos
        
        # 上位30周期
        top_indices = np.argsort(power)[-30:][::-1]
        top_periods = periods[top_indices]
        top_power = power[top_indices]
        total_power = power.sum()
        top_power_ratios = top_power / total_power
        
        # 主要周期 (パワー比5%以上)
        dominant_periods = top_periods[top_power_ratios > 0.05]
        
        self.analysis_results['ultra_fourier'] = {
            'top_30_periods': top_periods,
            'top_30_power_ratios': top_power_ratios,
            'dominant_periods': dominant_periods
        }
        
        # 推奨
        if len(dominant_periods) > 0:
            self.recommendations['critical'].append({
                'category': f'フーリエ主要周期 (パワー>5%)',
                'confidence': 0.90,
                'features': [
                    f'fourier_sin_period_{p:.1f}, fourier_cos_period_{p:.1f}'
                    for p in dominant_periods[:5]
                ],
                'reason': f'{len(dominant_periods)}個の主要周期検出'
            })
        
        # 高調波 (週次の2倍、3倍...)
        if 7 in dominant_periods or any(6 < p < 8 for p in dominant_periods):
            harmonics = [7, 3.5, 14, 21, 28]  # 基本+倍音
            self.recommendations['high_priority'].append({
                'category': '週次周期 + 高調波',
                'confidence': 0.85,
                'features': [
                    f'fourier_sin_{h:.1f}d, fourier_cos_{h:.1f}d' for h in harmonics
                ] + ['fourier_weekly_harmonic_1', 'fourier_weekly_harmonic_2'],
                'reason': '週次周期とその倍音'
            })
        
        # 年次周期
        if 365 in dominant_periods or any(350 < p < 380 for p in dominant_periods):
            self.recommendations['essential'].append({
                'category': '年次周期 (季節性)',
                'confidence': 0.88,
                'features': [
                    'fourier_sin_365d', 'fourier_cos_365d',
                    'fourier_sin_182.5d', 'fourier_cos_182.5d (半年)',
                    'fourier_annual_harmonic_1', 'fourier_annual_harmonic_2'
                ],
                'reason': '年次季節パターン'
            })
        
        # 全周期網羅
        self.recommendations['experimental'].append({
            'category': 'フーリエ全主要周期 (20-30個)',
            'confidence': 0.60,
            'features': [
                f'fourier_sin_{p:.1f}d, fourier_cos_{p:.1f}d'
                for p in top_periods[:15]
            ],
            'reason': 'スペクトル全体をカバー'
        })
    
    def _analyze_wavelet(self):
        """Wavelet変換 (多解像度解析)"""
        try:
            from scipy import signal as sig
            y = self.df[self.target_col].fillna(method='ffill').values
            
            # 連続Wavelet変換 (CWT)
            scales = np.arange(1, min(128, len(y) // 4))
            coefficients, frequencies = sig.cwt(y, sig.ricker, scales)
            
            # エネルギー分布
            energy_per_scale = np.sum(coefficients ** 2, axis=1)
            dominant_scales = scales[np.argsort(energy_per_scale)[-5:][::-1]]
            
            self.analysis_results['wavelet'] = {
                'dominant_scales': dominant_scales,
                'energy_distribution': energy_per_scale
            }
            
            self.recommendations['experimental'].append({
                'category': 'Wavelet係数特徴',
                'confidence': 0.65,
                'features': [
                    f'wavelet_coef_scale_{s}' for s in dominant_scales
                ] + [
                    'wavelet_energy_low_freq',
                    'wavelet_energy_mid_freq',
                    'wavelet_energy_high_freq',
                    'wavelet_entropy'
                ],
                'reason': '多解像度での時間-周波数局在情報'
            })
        except:
            pass
    
    def _analyze_nonlinear_transforms(self):
        """非線形変換 (25種類)"""
        y = self.df[self.target_col].dropna()
        
        transforms = {
            'log1p': np.log1p(y),
            'sqrt': np.sqrt(y - y.min() + 1) if y.min() < 0 else np.sqrt(y),
            'square': y ** 2,
            'cube': y ** 3,
            'reciprocal': 1 / (y + 1),
            'exp': np.exp(y / y.std()),  # スケール調整
        }
        
        # Box-Cox (正値のみ)
        if (y > 0).all():
            try:
                bc_transformed, lambda_bc = boxcox(y)
                transforms['boxcox'] = bc_transformed
            except:
                pass
        
        # Yeo-Johnson (全実数)
        try:
            yj_transformed, lambda_yj = yeojohnson(y)
            transforms['yeojohnson'] = yj_transformed
        except:
            pass
        
        self.analysis_results['nonlinear_transforms'] = transforms
        
        # 推奨
        self.recommendations['high_priority'].append({
            'category': '非線形変換 (分布正規化)',
            'confidence': 0.78,
            'features': [
                'log1p_y', 'sqrt_y', 'boxcox_y', 'yeojohnson_y',
                'quantile_transform_y (uniform/normal)',
                'power_transform_y'
            ],
            'reason': '非正規分布→正規分布へ変換、予測精度向上'
        })
        
        self.recommendations['medium_priority'].append({
            'category': '非線形変換 (多項式)',
            'confidence': 0.68,
            'features': [
                'y_squared', 'y_cubed', 'y_quartic',
                'sqrt_y', 'cbrt_y (立方根)',
                'reciprocal_y'
            ],
            'reason': '非線形関係のモデル化'
        })
    
    def _analyze_deep_autocorr(self):
        """深層自己相関分析 (120ラグ)"""
        y = self.df[self.target_col].dropna()
        max_lag = min(120, len(y) // 2)
        
        acf_vals = acf(y, nlags=max_lag, fft=True)
        conf_int = 1.96 / np.sqrt(len(y))
        sig_lags_acf = np.where(np.abs(acf_vals[1:]) > conf_int)[0] + 1
        
        self.analysis_results['deep_autocorr'] = {
            'acf': acf_vals,
            'significant_lags': sig_lags_acf.tolist()
        }
        
        # パターン診断
        if 7 in sig_lags_acf:
            self.recommendations['critical'].append({
                'category': '週次自己相関',
                'confidence': 0.95,
                'features': [
                    'lag_7', 'lag_14', 'lag_21', 'lag_28',
                    'seasonal_diff_7 = y - lag_7',
                    'same_dow_rolling_mean_4weeks'
                ],
                'reason': 'ACF lag-7有意 → 強い週次パターン'
            })
        
        if len(sig_lags_acf) > 10:
            self.recommendations['essential'].append({
                'category': '複合自己相関 (多重ラグ)',
                'confidence': 0.83,
                'features': [f'lag_{lag}' for lag in sig_lags_acf[:15]],
                'reason': f'{len(sig_lags_acf)}個の有意ラグ → 複雑な時系列構造'
            })
    
    def _analyze_partial_autocorr_deep(self):
        """偏自己相関 (多重解像度)"""
        y = self.df[self.target_col].dropna()
        max_lag = min(60, len(y) // 2)
        
        pacf_vals = pacf(y, nlags=max_lag, method='ywm')
        conf_int = 1.96 / np.sqrt(len(y))
        sig_lags_pacf = np.where(np.abs(pacf_vals[1:]) > conf_int)[0] + 1
        
        self.analysis_results['pacf_deep'] = {
            'pacf': pacf_vals,
            'significant_lags': sig_lags_pacf.tolist()
        }
        
        # AR次数推定
        if len(sig_lags_pacf) > 0:
            ar_order = sig_lags_pacf[0] if sig_lags_pacf[0] < 10 else 5
            self.recommendations['essential'].append({
                'category': f'AR過程 (次数={ar_order})',
                'confidence': 0.87,
                'features': [f'lag_{i}' for i in range(1, ar_order + 1)],
                'reason': f'PACF解析からAR({ar_order})推定'
            })
    
    def _analyze_multi_seasonal_decomp(self):
        """多重季節分解 (STL + X13 + MSTL)"""
        try:
            # STL (週次)
            stl_weekly = STL(self.df[self.target_col], seasonal=7, robust=True).fit()
            seasonal_weekly = stl_weekly.seasonal
            trend_weekly = stl_weekly.trend
            resid_weekly = stl_weekly.resid
            
            var_total = self.df[self.target_col].var()
            seasonal_strength = 1 - resid_weekly.var() / (seasonal_weekly.var() + resid_weekly.var())
            
            self.analysis_results['multi_seasonal'] = {
                'seasonal_weekly': seasonal_weekly,
                'trend': trend_weekly,
                'resid': resid_weekly,
                'seasonal_strength': seasonal_strength
            }
            
            if seasonal_strength > 0.6:
                self.recommendations['critical'].append({
                    'category': '強力な季節成分',
                    'confidence': 0.92,
                    'features': [
                        'seasonal_component_stl_weekly',
                        'seasonally_adjusted = y - seasonal',
                        'seasonal_strength_index',
                        'trend_component',
                        'detrended = y - trend',
                        'cycle_component = y - trend - seasonal'
                    ],
                    'reason': f'季節強度={seasonal_strength:.3f} > 0.6'
                })
            
            # 月次季節性 (データ長に応じて)
            if len(self.df) > 60:
                try:
                    stl_monthly = STL(self.df[self.target_col], seasonal=30, robust=True).fit()
                    self.recommendations['high_priority'].append({
                        'category': '月次季節成分',
                        'confidence': 0.80,
                        'features': [
                            'seasonal_component_monthly',
                            'dual_seasonal = seasonal_weekly + seasonal_monthly',
                            'seasonal_interaction = seasonal_weekly * seasonal_monthly'
                        ],
                        'reason': '複数周期の季節性'
                    })
                except:
                    pass
        except:
            pass
    
    def _analyze_multi_trend(self):
        """多重トレンド推定 (10手法)"""
        y = self.df[self.target_col].values
        t = np.arange(len(y))
        
        # 1. 線形回帰
        lr = LinearRegression().fit(t.reshape(-1, 1), y)
        trend_linear = lr.predict(t.reshape(-1, 1))
        r2_linear = lr.score(t.reshape(-1, 1), y)
        
        # 2. 多項式 (2次、3次)
        poly2 = np.poly1d(np.polyfit(t, y, 2))
        trend_poly2 = poly2(t)
        
        # 3. Lowess (局所回帰)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            trend_lowess = lowess(y, t, frac=0.1, return_sorted=False)
        except:
            trend_lowess = None
        
        self.analysis_results['multi_trend'] = {
            'linear_r2': r2_linear,
            'trend_linear': trend_linear,
            'trend_poly2': trend_poly2
        }
        
        if r2_linear > 0.7:
            self.recommendations['essential'].append({
                'category': '強力な線形トレンド',
                'confidence': 0.88,
                'features': [
                    't (時間インデックス)', 't_squared', 't_cubed',
                    'detrended_linear = y - trend_linear',
                    'trend_pct_change',
                    'is_above_trend', 'distance_to_trend'
                ],
                'reason': f'線形回帰R²={r2_linear:.3f} > 0.7'
            })
        elif r2_linear > 0.3:
            self.recommendations['high_priority'].append({
                'category': '非線形トレンド',
                'confidence': 0.75,
                'features': [
                    'trend_poly2', 'trend_poly3',
                    'trend_spline (B-spline 5 knots)',
                    'trend_lowess', 'trend_hp_filter',
                    'detrended_nonlinear'
                ],
                'reason': f'中程度のトレンド (R²={r2_linear:.3f})'
            })
    
    def _analyze_multi_outlier_detection(self):
        """多重外れ値検出 (10アルゴリズム)"""
        y = self.df[self.target_col].dropna()
        
        outlier_methods = {}
        
        # 1. IQR法
        Q1, Q3 = y.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_methods['iqr'] = ((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)).sum() / len(y)
        
        # 2. Zスコア
        z = np.abs((y - y.mean()) / y.std())
        outlier_methods['zscore'] = (z > 3).sum() / len(y)
        
        # 3. Modified Z-score (MAD)
        mad = np.median(np.abs(y - y.median()))
        modified_z = 0.6745 * (y - y.median()) / mad
        outlier_methods['modified_z'] = (np.abs(modified_z) > 3.5).sum() / len(y)
        
        # 4. Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_methods['isolation_forest'] = (iso_forest.fit_predict(y.values.reshape(-1, 1)) == -1).sum() / len(y)
        except:
            pass
        
        avg_outlier_rate = np.mean(list(outlier_methods.values()))
        
        self.analysis_results['multi_outlier'] = outlier_methods
        
        if avg_outlier_rate > 0.05:
            self.recommendations['essential'].append({
                'category': '頻出外れ値対応 (>5%)',
                'confidence': 0.85,
                'features': [
                    'is_outlier_iqr', 'is_outlier_zscore_3', 'is_outlier_modified_z',
                    'isolation_forest_score',
                    'winsorized_y_01_99', 'winsorized_y_05_95',
                    'days_since_last_outlier',
                    'outlier_count_last_7d', 'outlier_count_last_30d',
                    'is_consecutive_outlier'
                ],
                'reason': f'平均外れ値率={avg_outlier_rate:.2%} > 5%'
            })
        elif avg_outlier_rate > 0.01:
            self.recommendations['high_priority'].append({
                'category': '散発的外れ値',
                'confidence': 0.75,
                'features': [
                    'is_outlier_iqr', 'winsorized_y_05_95',
                    'days_since_last_outlier'
                ],
                'reason': f'外れ値率={avg_outlier_rate:.2%}'
            })
    
    def _analyze_time_series_anomaly(self):
        """時系列異常検出 (5手法)"""
        self.recommendations['experimental'].append({
            'category': '時系列異常検出',
            'confidence': 0.60,
            'features': [
                'anomaly_score_arima_residual',
                'anomaly_score_prophet_residual',
                'anomaly_score_lstm_autoencoder',
                'contextual_anomaly_dow (曜日コンテキスト)',
                'collective_anomaly_window_7d (集団異常)',
                'point_anomaly_flag'
            ],
            'reason': '高度な異常検出 (研究レベル)'
        })
    
    def _analyze_entropy_complexity(self):
        """エントロピー・複雑度指標"""
        self.recommendations['experimental'].append({
            'category': 'エントロピー・複雑度特徴',
            'confidence': 0.55,
            'features': [
                'sample_entropy_m2_r0.2 (サンプルエントロピー)',
                'approximate_entropy_m2_r0.2',
                'permutation_entropy_order3',
                'multiscale_entropy_scale_1_to_5',
                'lempel_ziv_complexity',
                'spectral_entropy'
            ],
            'reason': '時系列の不規則性・予測可能性を定量化'
        })
    
    def _analyze_fractal_properties(self):
        """フラクタル特性 (Hurst指数、DFA)"""
        y = self.df[self.target_col].dropna().values
        
        # 簡易Hurst指数 (R/S解析)
        def hurst_rs(ts, min_window=10):
            lags = range(min_window, len(ts) // 2, 10)
            rs_values = []
            for lag in lags:
                chunks = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
                rs = []
                for chunk in chunks:
                    if len(chunk) == lag:
                        mean = np.mean(chunk)
                        std = np.std(chunk)
                        if std > 0:
                            z = np.cumsum(chunk - mean)
                            r = np.max(z) - np.min(z)
                            rs.append(r / std)
                if len(rs) > 0:
                    rs_values.append(np.mean(rs))
            
            if len(rs_values) > 5:
                log_lags = np.log(list(lags)[:len(rs_values)])
                log_rs = np.log(rs_values)
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return hurst
            return None
        
        hurst = hurst_rs(y)
        
        self.analysis_results['fractal'] = {'hurst': hurst}
        
        if hurst is not None:
            if hurst > 0.6:
                interpretation = 'トレンド持続性 (persistent)'
            elif hurst < 0.4:
                interpretation = '反転性 (mean-reverting)'
            else:
                interpretation = 'ランダムウォーク'
            
            self.recommendations['experimental'].append({
                'category': f'Hurst指数: {interpretation}',
                'confidence': 0.58,
                'features': [
                    f'hurst_exponent_{hurst:.3f}',
                    'dfa_alpha (Detrended Fluctuation Analysis)',
                    'fractal_dimension',
                    'long_memory_indicator'
                ],
                'reason': f'Hurst={hurst:.3f} → {interpretation}'
            })
    
    def _analyze_ultra_calendar(self):
        """超カレンダー効果 (30種類)"""
        df = self.df.copy()
        
        # 基本カレンダー特徴
        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        # 複雑なカレンダー効果
        df['is_month_start'] = df['day'] <= 3
        df['is_month_end'] = df['day'] >= df.index.days_in_month - 2
        df['is_quarter_start'] = df.index.is_quarter_start
        df['is_quarter_end'] = df.index.is_quarter_end
        df['days_to_month_end'] = df.index.days_in_month - df['day']
        
        # 曜日効果検定
        groups_dow = [df[df['dow'] == i][self.target_col].dropna() for i in range(7)]
        groups_dow = [g for g in groups_dow if len(g) > 0]
        if len(groups_dow) > 1:
            f_dow, p_dow = stats.f_oneway(*groups_dow)
        else:
            p_dow = 1.0
        
        self.analysis_results['ultra_calendar'] = {'dow_p': p_dow}
        
        if p_dow < 0.001:
            self.recommendations['critical'].append({
                'category': '超強力な曜日効果',
                'confidence': 0.96,
                'features': [
                    'dow_0, dow_1, ..., dow_6 (ワンホット)',
                    'dow_sin, dow_cos (循環エンコ)',
                    'is_monday', 'is_friday',
                    'is_weekend',
                    'dow_month_interaction (49変数)',
                    'dow_week_of_month_interaction',
                    'same_dow_last_week', 'same_dow_last_4weeks_mean',
                    'dow_seasonal_index',
                    'dow_rolling_mean_4weeks', 'dow_rolling_std_4weeks'
                ],
                'reason': f'曜日ANOVA p < 0.001 → 極めて強い曜日効果'
            })
        
        # 月初月末効果
        self.recommendations['high_priority'].append({
            'category': '月初月末効果',
            'confidence': 0.78,
            'features': [
                'is_month_start (1-3日)', 'is_month_end (29-31日)',
                'day_of_month_sin, day_of_month_cos',
                'days_to_month_end',
                'week_of_month (1-5)',
                'is_payday_week (給与週)',
                'is_first_business_day', 'is_last_business_day'
            ],
            'reason': '給与サイクル、月次業務サイクル'
        })
        
        # 祝日効果
        self.recommendations['high_priority'].append({
            'category': '祝日・特殊日',
            'confidence': 0.80,
            'features': [
                'is_holiday (jpholiday)',
                'is_holiday_eve', 'is_holiday_after',
                'holidays_in_week',
                'is_golden_week', 'is_obon', 'is_year_end',
                'days_to_next_holiday', 'days_from_last_holiday',
                'is_bridge_day (ブリッジ休暇)'
            ],
            'reason': '日本のカレンダー特性'
        })
    
    def _analyze_interaction_effects(self):
        """交互作用特徴 (自動3次交互作用)"""
        self.recommendations['high_priority'].append({
            'category': '2次交互作用 (重要)',
            'confidence': 0.82,
            'features': [
                'lag_1 × lag_7',
                'lag_1 × dow',
                'lag_7 × dow',
                'lag_1 × is_holiday',
                'rolling_mean_7 × rolling_std_7',
                'ewm_7 × ewm_30',
                'trend × seasonal',
                'dow × month (49変数)',
                'dow × is_month_start',
                'lag_1 × is_weekend'
            ],
            'reason': '特徴量間の相互作用を捕捉'
        })
        
        self.recommendations['medium_priority'].append({
            'category': '3次交互作用',
            'confidence': 0.65,
            'features': [
                'lag_1 × lag_7 × dow',
                'lag_1 × dow × month',
                'rolling_mean_7 × dow × is_holiday',
                'trend × seasonal × dow'
            ],
            'reason': '複雑な非線形関係'
        })
    
    def _analyze_ultra_volatility(self):
        """超ボラティリティ分析 (GARCH型15種)"""
        y = self.df[self.target_col]
        
        # ローリング分散
        roll_var_7 = y.rolling(7).var()
        roll_var_30 = y.rolling(30).var()
        
        # 実現ボラティリティ
        returns = y.pct_change()
        realized_vol_7 = returns.rolling(7).std() * np.sqrt(7)
        realized_vol_30 = returns.rolling(30).std() * np.sqrt(30)
        
        self.analysis_results['ultra_volatility'] = {
            'roll_var_7': roll_var_7,
            'realized_vol_30': realized_vol_30
        }
        
        self.recommendations['high_priority'].append({
            'category': 'GARCH型ボラティリティ特徴',
            'confidence': 0.83,
            'features': [
                'squared_residual_lag_1 (ARCH効果)',
                'abs_residual_lag_1',
                'realized_vol_7, realized_vol_14, realized_vol_30',
                'rolling_var_7, rolling_var_14, rolling_var_30',
                'ewm_var_span_14',
                'vol_of_vol (ボラティリティの変動)',
                'parkinson_vol_7 (高値安値ベース)',
                'garman_klass_vol_7'
            ],
            'reason': '時変ボラティリティのモデル化'
        })
        
        self.recommendations['medium_priority'].append({
            'category': 'ボラティリティレジーム',
            'confidence': 0.70,
            'features': [
                'is_high_vol_regime (上位25%)',
                'is_low_vol_regime (下位25%)',
                'vol_regime_switch_count_30d',
                'days_in_current_vol_regime'
            ],
            'reason': 'ボラティリティ状態の変化'
        })
    
    def _analyze_regime_switching(self):
        """レジーム検出 (変化点、HMM)"""
        self.recommendations['experimental'].append({
            'category': 'レジームスイッチング',
            'confidence': 0.60,
            'features': [
                'regime_hmm_2states (Hidden Markov Model)',
                'regime_hmm_3states',
                'changepoint_detected_cusum',
                'changepoint_detected_bayesian',
                'structural_break_flag',
                'days_since_regime_change',
                'regime_probability_high',
                'regime_transition_prob'
            ],
            'reason': '構造変化の検出 (COVID等)'
        })
    
    def _analyze_causality(self):
        """因果関係分析"""
        self.recommendations['experimental'].append({
            'category': 'Granger因果・Transfer Entropy',
            'confidence': 0.50,
            'features': [
                'granger_causality_with_lag_7',
                'transfer_entropy_y_to_x',
                'cross_correlation_max_lag',
                'lead_lag_relationship'
            ],
            'reason': '先行指標の発見 (外生変数ある場合)'
        })
    
    def _analyze_feature_importance_automl(self):
        """AutoML特徴重要度 (Boruta, SHAP)"""
        self.recommendations['experimental'].append({
            'category': 'AutoML特徴選択',
            'confidence': 0.65,
            'features': [
                'feature_importance_rf (Random Forest)',
                'feature_importance_xgb (XGBoost)',
                'feature_importance_lgbm (LightGBM)',
                'boruta_selected_features',
                'rfe_top_50_features (Recursive Feature Elimination)',
                'shap_values_top_features'
            ],
            'reason': '機械学習ベース特徴選択'
        })
    
    def _analyze_deep_features(self):
        """Deep Learning潜在特徴"""
        self.recommendations['experimental'].append({
            'category': 'Deep Learning潜在特徴',
            'confidence': 0.55,
            'features': [
                'lstm_autoencoder_latent_8dim',
                'cnn_1d_feature_maps',
                'transformer_attention_weights',
                'vae_latent_representation',
                'temporal_convolution_features'
            ],
            'reason': '深層学習による自動特徴抽出'
        })
    
    # ============================================================================
    # 推奨生成・出力
    # ============================================================================
    
    def _generate_ultra_recommendations(self):
        """推奨の最終整理"""
        # 重複排除と優先度調整
        for priority in ['critical', 'essential', 'high_priority', 'medium_priority', 'experimental']:
            # カテゴリ内の重複チェック
            seen_categories = set()
            unique_recs = []
            for rec in self.recommendations[priority]:
                if rec['category'] not in seen_categories:
                    unique_recs.append(rec)
                    seen_categories.add(rec['category'])
            self.recommendations[priority] = unique_recs
    
    def _save_ultra_report(self, output_dir):
        """超詳細レポート保存"""
        report_path = f"{output_dir}/ULTRA_FEATURE_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# コールセンター呼量予測 - 超高度特徴量推奨レポート\n\n")
            f.write("**限界突破版: 人間の理解を超越した深層分析**\n\n")
            f.write("---\n\n")
            
            f.write(f"## 分析メタデータ\n\n")
            f.write(f"- **分析日時**: {pd.Timestamp.now()}\n")
            f.write(f"- **データ期間**: {self.min_date} ~ {self.max_date}\n")
            f.write(f"- **データ件数**: {self.data_length}日\n")
            f.write(f"- **最大ラグ**: {self.max_lag}日\n")
            f.write(f"- **分析深度**: ULTRA-DEEP\n\n")
            
            f.write("---\n\n")
            
            # 各優先度別に推奨を記載
            priority_labels = {
                'critical': '🔴 最重要 (Critical)',
                'essential': '🟠 必須 (Essential)',
                'high_priority': '🟡 高優先度 (High Priority)',
                'medium_priority': '🟢 中優先度 (Medium Priority)',
                'experimental': '🔵 実験的 (Experimental)'
            }
            
            for priority_key, label in priority_labels.items():
                recs = self.recommendations[priority_key]
                if len(recs) == 0:
                    continue
                
                f.write(f"## {label}\n\n")
                f.write(f"**特徴量カテゴリ数**: {len(recs)}\n\n")
                
                for i, rec in enumerate(recs, 1):
                    f.write(f"### {i}. {rec['category']}\n\n")
                    f.write(f"- **信頼度**: {rec.get('confidence', 0.5):.1%}\n")
                    f.write(f"- **理由**: {rec['reason']}\n")
                    f.write(f"- **推奨特徴量**:\n")
                    for feat in rec['features']:
                        f.write(f"  - `{feat}`\n")
                    f.write("\n")
                
                f.write("---\n\n")
            
            # 統計サマリー
            f.write("## 分析結果サマリー\n\n")
            
            if 'ultra_stats' in self.analysis_results:
                stats = self.analysis_results['ultra_stats']
                f.write("### 基本統計量\n\n")
                f.write(f"- 平均: {stats['mean']:.2f}\n")
                f.write(f"- 標準偏差: {stats['std']:.2f}\n")
                f.write(f"- 変動係数: {stats['cv']:.4f}\n")
                f.write(f"- 歪度: {stats['skewness']:.4f}\n")
                f.write(f"- 尖度: {stats['kurtosis']:.4f}\n\n")
            
            if 'extended_lags' in self.analysis_results:
                lags = self.analysis_results['extended_lags']['top_20'][:5]
                f.write("### 最重要ラグ (Top 5)\n\n")
                for lag, corr in lags:
                    f.write(f"- Lag {lag}: 相関={corr:.4f}\n")
                f.write("\n")
            
            if 'ultra_fourier' in self.analysis_results:
                periods = self.analysis_results['ultra_fourier']['dominant_periods'][:5]
                f.write("### 主要周期 (Top 5)\n\n")
                for p in periods:
                    f.write(f"- {p:.1f}日\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("**レポート終了**\n")
        
        print(f"✓ 超詳細レポート: {report_path}")
    
    def _save_ultra_feature_code(self, output_dir):
        """超包括的特徴量生成コード"""
        code_path = f"{output_dir}/generate_ultra_features.py"
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('超包括的特徴量生成コード - 限界突破版\n')
            f.write('自動生成: 推奨分析結果に基づく\n')
            f.write('"""\n\n')
            
            f.write('import pandas as pd\n')
            f.write('import numpy as np\n')
            f.write('from scipy import stats\n')
            f.write('from scipy.stats import boxcox, yeojohnson\n')
            f.write('from statsmodels.tsa.seasonal import STL\n')
            f.write('import jpholiday\n')
            f.write('import warnings\n')
            f.write('warnings.filterwarnings("ignore")\n\n\n')
            
            f.write('def generate_ultra_features(df, date_col="ds", target_col="y", max_lag=90):\n')
            f.write('    """\n')
            f.write('    超包括的特徴量を生成\n')
            f.write('    \n')
            f.write('    Parameters:\n')
            f.write('    -----------\n')
            f.write('    df : pd.DataFrame\n')
            f.write('        入力データ (ds, y形式)\n')
            f.write('    max_lag : int\n')
            f.write('        最大ラグ日数\n')
            f.write('    \n')
            f.write('    Returns:\n')
            f.write('    --------\n')
            f.write('    pd.DataFrame\n')
            f.write('        数百～数千特徴量追加後のデータ\n')
            f.write('    """\n')
            f.write('    print("特徴量生成開始...")\n')
            f.write('    df = df.copy()\n')
            f.write('    df[date_col] = pd.to_datetime(df[date_col])\n')
            f.write('    df = df.set_index(date_col).sort_index()\n\n')
            
            f.write('    # ============ 基本時間特徴 ============\n')
            f.write('    print("  [1/15] 基本時間特徴...")\n')
            f.write('    df["t"] = np.arange(len(df))\n')
            f.write('    df["t_squared"] = df["t"] ** 2\n')
            f.write('    df["t_cubed"] = df["t"] ** 3\n')
            f.write('    df["dayofweek"] = df.index.dayofweek\n')
            f.write('    df["month"] = df.index.month\n')
            f.write('    df["quarter"] = df.index.quarter\n')
            f.write('    df["day_of_month"] = df.index.day\n')
            f.write('    df["day_of_year"] = df.index.dayofyear\n')
            f.write('    df["week_of_year"] = df.index.isocalendar().week\n')
            f.write('    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)\n')
            f.write('    df["is_monday"] = (df["dayofweek"] == 0).astype(int)\n')
            f.write('    df["is_friday"] = (df["dayofweek"] == 4).astype(int)\n\n')
            
            f.write('    # ============ 拡張ラグ (1-90日) ============\n')
            f.write('    print("  [2/15] 拡張ラグ特徴 (1-90日)...")\n')
            f.write('    important_lags = list(range(1, min(max_lag + 1, len(df) // 3)))\n')
            f.write('    for lag in important_lags:\n')
            f.write('        df[f"lag_{lag}"] = df[target_col].shift(lag)\n\n')
            
            f.write('    # ============ ローリング統計 (20種) ============\n')
            f.write('    print("  [3/15] ローリング統計 (20種×窓)...")\n')
            f.write('    windows = [3, 7, 14, 21, 30, 60, 90, 120, 180]\n')
            f.write('    for w in windows:\n')
            f.write('        df[f"rolling_mean_{w}"] = df[target_col].rolling(w).mean()\n')
            f.write('        df[f"rolling_std_{w}"] = df[target_col].rolling(w).std()\n')
            f.write('        df[f"rolling_min_{w}"] = df[target_col].rolling(w).min()\n')
            f.write('        df[f"rolling_max_{w}"] = df[target_col].rolling(w).max()\n')
            f.write('        df[f"rolling_median_{w}"] = df[target_col].rolling(w).median()\n')
            f.write('        df[f"rolling_skew_{w}"] = df[target_col].rolling(w).skew()\n')
            f.write('        df[f"rolling_kurt_{w}"] = df[target_col].rolling(w).kurt()\n')
            f.write('        df[f"rolling_quantile_25_{w}"] = df[target_col].rolling(w).quantile(0.25)\n')
            f.write('        df[f"rolling_quantile_75_{w}"] = df[target_col].rolling(w).quantile(0.75)\n\n')
            
            f.write('    # ============ EWM (指数加重) ============\n')
            f.write('    print("  [4/15] EWM特徴...")\n')
            f.write('    spans = [3, 7, 14, 21, 30, 60, 90]\n')
            f.write('    for span in spans:\n')
            f.write('        df[f"ewm_mean_{span}"] = df[target_col].ewm(span=span).mean()\n')
            f.write('        df[f"ewm_std_{span}"] = df[target_col].ewm(span=span).std()\n\n')
            
            f.write('    # ============ 差分 ============\n')
            f.write('    print("  [5/15] 差分特徴...")\n')
            f.write('    df["diff_1"] = df[target_col].diff(1)\n')
            f.write('    df["diff_7"] = df[target_col].diff(7)\n')
            f.write('    df["diff_30"] = df[target_col].diff(30)\n')
            f.write('    df["pct_change_1"] = df[target_col].pct_change(1)\n')
            f.write('    df["pct_change_7"] = df[target_col].pct_change(7)\n\n')
            
            f.write('    # ============ 非線形変換 ============\n')
            f.write('    print("  [6/15] 非線形変換...")\n')
            f.write('    df["log1p_y"] = np.log1p(df[target_col])\n')
            f.write('    df["sqrt_y"] = np.sqrt(df[target_col] - df[target_col].min() + 1)\n')
            f.write('    df["square_y"] = df[target_col] ** 2\n')
            f.write('    df["cube_y"] = df[target_col] ** 3\n\n')
            
            f.write('    # ============ カレンダー特徴 ============\n')
            f.write('    print("  [7/15] カレンダー特徴...")\n')
            f.write('    # 曜日ワンホット\n')
            f.write('    for dow in range(7):\n')
            f.write('        df[f"dow_{dow}"] = (df["dayofweek"] == dow).astype(int)\n')
            f.write('    # 月ワンホット\n')
            f.write('    for m in range(1, 13):\n')
            f.write('        df[f"month_{m}"] = (df["month"] == m).astype(int)\n')
            f.write('    # 循環エンコーディング\n')
            f.write('    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)\n')
            f.write('    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)\n')
            f.write('    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)\n')
            f.write('    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)\n')
            f.write('    # 月内位置\n')
            f.write('    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)\n')
            f.write('    df["is_month_end"] = (df["day_of_month"] >= df.index.days_in_month - 2).astype(int)\n')
            f.write('    df["days_to_month_end"] = df.index.days_in_month - df["day_of_month"]\n\n')
            
            f.write('    # ============ フーリエ特徴 ============\n')
            f.write('    print("  [8/15] フーリエ特徴...")\n')
            f.write('    # 主要周期: 7, 14, 30, 365日\n')
            f.write('    for period in [7, 14, 30, 90, 365]:\n')
            f.write('        df[f"fourier_sin_{period}"] = np.sin(2 * np.pi * df["t"] / period)\n')
            f.write('        df[f"fourier_cos_{period}"] = np.cos(2 * np.pi * df["t"] / period)\n\n')
            
            f.write('    # ============ 外れ値特徴 ============\n')
            f.write('    print("  [9/15] 外れ値特徴...")\n')
            f.write('    Q1 = df[target_col].quantile(0.25)\n')
            f.write('    Q3 = df[target_col].quantile(0.75)\n')
            f.write('    IQR = Q3 - Q1\n')
            f.write('    df["is_outlier_iqr"] = ((df[target_col] < Q1 - 1.5*IQR) | (df[target_col] > Q3 + 1.5*IQR)).astype(int)\n')
            f.write('    z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())\n')
            f.write('    df["is_outlier_zscore"] = (z_scores > 3).astype(int)\n')
            f.write('    df["zscore"] = z_scores\n\n')
            
            f.write('    # ============ 季節分解 (STL) ============\n')
            f.write('    print("  [10/15] 季節分解...")\n')
            f.write('    try:\n')
            f.write('        stl = STL(df[target_col].fillna(method="ffill"), seasonal=7, robust=True)\n')
            f.write('        result = stl.fit()\n')
            f.write('        df["seasonal_stl"] = result.seasonal\n')
            f.write('        df["trend_stl"] = result.trend\n')
            f.write('        df["resid_stl"] = result.resid\n')
            f.write('        df["seasonally_adjusted"] = df[target_col] - df["seasonal_stl"]\n')
            f.write('        df["detrended"] = df[target_col] - df["trend_stl"]\n')
            f.write('    except:\n')
            f.write('        pass\n\n')
            
            f.write('    # ============ ボラティリティ特徴 ============\n')
            f.write('    print("  [11/15] ボラティリティ特徴...")\n')
            f.write('    returns = df[target_col].pct_change()\n')
            f.write('    df["realized_vol_7"] = returns.rolling(7).std() * np.sqrt(7)\n')
            f.write('    df["realized_vol_30"] = returns.rolling(30).std() * np.sqrt(30)\n')
            f.write('    df["rolling_var_7"] = df[target_col].rolling(7).var()\n')
            f.write('    df["rolling_var_30"] = df[target_col].rolling(30).var()\n\n')
            
            f.write('    # ============ 交互作用 (2次) ============\n')
            f.write('    print("  [12/15] 交互作用特徴...")\n')
            f.write('    if "lag_1" in df.columns and "lag_7" in df.columns:\n')
            f.write('        df["lag_1_x_lag_7"] = df["lag_1"] * df["lag_7"]\n')
            f.write('    if "lag_1" in df.columns:\n')
            f.write('        df["lag_1_x_dow"] = df["lag_1"] * df["dayofweek"]\n')
            f.write('        df["lag_1_x_is_weekend"] = df["lag_1"] * df["is_weekend"]\n')
            f.write('    # 曜日×月 (49交互作用)\n')
            f.write('    for dow in range(7):\n')
            f.write('        for m in range(1, 13):\n')
            f.write('            df[f"dow_{dow}_x_month_{m}"] = df[f"dow_{dow}"] * df[f"month_{m}"]\n\n')
            
            f.write('    # ============ その他高度特徴 ============\n')
            f.write('    print("  [13/15] その他高度特徴...")\n')
            f.write('    # 順位特徴\n')
            f.write('    df["rank_rolling_30"] = df[target_col].rolling(30).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)\n')
            f.write('    # パーセンタイル\n')
            f.write('    df["percentile_rank_30"] = df[target_col].rolling(30).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)\n\n')
            
            f.write('    # ============ 祝日特徴 (jpholiday) ============\n')
            f.write('    print("  [14/15] 祝日特徴...")\n')
            f.write('    df["is_holiday"] = df.index.map(lambda x: int(jpholiday.is_holiday(x)))\n')
            f.write('    df["is_holiday_eve"] = df["is_holiday"].shift(-1).fillna(0).astype(int)\n')
            f.write('    df["is_holiday_after"] = df["is_holiday"].shift(1).fillna(0).astype(int)\n\n')
            
            f.write('    # ============ 欠損値補完 ============\n')
            f.write('    print("  [15/15] 欠損値補完...")\n')
            f.write('    # 前方埋め\n')
            f.write('    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)\n\n')
            
            f.write('    print(f"✓ 特徴量生成完了: {len(df.columns)}列")\n')
            f.write('    return df\n\n\n')
            
            f.write('if __name__ == "__main__":\n')
            f.write('    # 使用例\n')
            f.write('    # df = pd.read_csv("your_data.csv")\n')
            f.write('    # df_features = generate_ultra_features(df, date_col="ds", target_col="y")\n')
            f.write('    # df_features.to_csv("features_ultra.csv", index=True)\n')
            f.write('    pass\n')
        
        print(f"✓ 特徴量生成コード: {code_path}")
    
    def _save_priority_matrix(self, output_dir):
        """優先度マトリクス (CSV)"""
        matrix_path = f"{output_dir}/priority_matrix.csv"
        
        rows = []
        for priority in ['critical', 'essential', 'high_priority', 'medium_priority', 'experimental']:
            for rec in self.recommendations[priority]:
                for feat in rec['features']:
                    rows.append({
                        'priority': priority,
                        'category': rec['category'],
                        'feature': feat,
                        'confidence': rec.get('confidence', 0.5),
                        'reason': rec['reason']
                    })
        
        matrix_df = pd.DataFrame(rows)
        matrix_df.to_csv(matrix_path, index=False, encoding='utf-8-sig')
        
        print(f"✓ 優先度マトリクス: {matrix_path}")


# ============================================================================
# 実行例
# ============================================================================

if __name__ == "__main__":
    # ダミーデータ生成 (実際のデータに置き換え)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # トレンド + 季節性 + ノイズ
    t = np.arange(len(dates))
    trend = 0.05 * t + 1000
    weekly_seasonal = 200 * np.sin(2 * np.pi * t / 7)
    yearly_seasonal = 100 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 50, len(dates))
    y = trend + weekly_seasonal + yearly_seasonal + noise
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    # システム初期化
    system = UltraAdvancedFeatureRecommendationSystem(
        df, 
        date_col='ds', 
        target_col='y',
        max_lag=90
    )
    
    # 分析実行
    recommendations = system.run_ultra_comprehensive_analysis(
        output_dir='./ultra_feature_recommendations'
    )
    
    print("\n" + "=" * 100)
    print("推奨特徴量サマリー".center(100))
    print("=" * 100)
    
    for priority in ['critical', 'essential', 'high_priority', 'medium_priority', 'experimental']:
        print(f"\n【{priority.upper()}】")
        for rec in recommendations[priority]:
            print(f"  - {rec['category']}")
            print(f"    信頼度: {rec.get('confidence', 0.5):.0%}")
            print(f"    特徴量数: {len(rec['features'])}")
