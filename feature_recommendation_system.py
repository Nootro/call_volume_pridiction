"""
コールセンター呼量予測 - 特徴量推奨システム
データを分析し、最適な特徴量を自動提案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FeatureRecommendationSystem:
    """
    データを分析し、推奨特徴量を自動生成するシステム
    """
    
    def __init__(self, df, date_col='ds', target_col='y'):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            入力データ (ds, y 形式)
        date_col : str
            日付カラム名
        target_col : str
            目的変数カラム名
        """
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        
        # 日付変換とソート
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.set_index(date_col).sort_index()
        self.df = self.df[[target_col]]
        
        # 分析結果格納
        self.analysis_results = {}
        self.recommendations = {
            'essential': [],      # 必須特徴量
            'high_priority': [],  # 高優先度
            'medium_priority': [], # 中優先度
            'optional': []        # オプション
        }
        self.feature_code_snippets = []
        
    def run_complete_analysis(self, output_dir='./feature_recommendations'):
        """全分析を実行し推奨特徴量を生成"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("コールセンター呼量予測 - 特徴量推奨システム")
        print("=" * 80)
        print(f"\nデータ期間: {self.df.index.min()} ~ {self.df.index.max()}")
        print(f"データ件数: {len(self.df)} 日")
        print(f"目的変数: {self.target_col}")
        print("\n" + "=" * 80)
        
        # 各種分析実行
        print("\n[1/12] 基本統計量分析...")
        self._analyze_basic_statistics()
        
        print("[2/12] トレンド分析...")
        self._analyze_trend()
        
        print("[3/12] 周期性分析...")
        self._analyze_periodicity()
        
        print("[4/12] 自己相関分析...")
        self._analyze_autocorrelation()
        
        print("[5/12] スペクトル解析...")
        self._analyze_spectrum()
        
        print("[6/12] 定常性検定...")
        self._analyze_stationarity()
        
        print("[7/12] 季節分解...")
        self._analyze_seasonal_decomposition()
        
        print("[8/12] 異常値分析...")
        self._analyze_outliers()
        
        print("[9/12] カレンダー効果分析...")
        self._analyze_calendar_effects()
        
        print("[10/12] 変動性分析...")
        self._analyze_volatility()
        
        print("[11/12] ラグ重要度分析...")
        self._analyze_lag_importance()
        
        print("[12/12] 特徴量推奨レポート生成...")
        self._generate_recommendations()
        
        # レポート出力
        self._save_detailed_report(output_dir)
        self._save_feature_generation_code(output_dir)
        self._create_visualization_dashboard(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ 分析完了!")
        print(f"✓ 推奨特徴量: 必須 {len(self.recommendations['essential'])}, "
              f"高優先度 {len(self.recommendations['high_priority'])}, "
              f"中優先度 {len(self.recommendations['medium_priority'])}, "
              f"オプション {len(self.recommendations['optional'])}")
        print(f"✓ レポート保存先: {output_dir}")
        print("=" * 80)
        
        return self.recommendations
    
    def _analyze_basic_statistics(self):
        """基本統計量分析"""
        y = self.df[self.target_col]
        
        stats_dict = {
            'mean': y.mean(),
            'std': y.std(),
            'cv': y.std() / y.mean(),
            'skewness': y.skew(),
            'kurtosis': y.kurtosis(),
            'min': y.min(),
            'max': y.max(),
            'q25': y.quantile(0.25),
            'q50': y.quantile(0.50),
            'q75': y.quantile(0.75)
        }
        
        self.analysis_results['basic_stats'] = stats_dict
        
        # CV診断
        cv = stats_dict['cv']
        if cv < 0.15:
            self.recommendations['high_priority'].append({
                'category': '時間特徴',
                'reason': f'CV={cv:.3f} (低変動) → 決定論的パターンが支配的',
                'features': ['曜日ダミー', '月ダミー', '祝日フラグ', '週内位置', '月初月末フラグ']
            })
        elif cv < 0.30:
            self.recommendations['high_priority'].append({
                'category': 'ラグ+ローリング特徴',
                'reason': f'CV={cv:.3f} (中変動) → ラグと短期統計が有効',
                'features': ['lag_1, lag_7, lag_14', 'rolling_mean_7/14/30', 'ewm_7/14', '曜日×月交互作用']
            })
        else:
            self.recommendations['essential'].append({
                'category': 'GARCH型+外生変数',
                'reason': f'CV={cv:.3f} (高変動) → 条件付き分散モデル化必須',
                'features': ['squared_residual_lag_1', 'realized_vol_7/30', 'Box-Cox変換', 'Zスコア(ローリング)', '外生変数(プロモーション等)']
            })
        
        # 歪度診断
        skew = stats_dict['skewness']
        if abs(skew) > 0.5:
            if skew > 0:
                self.recommendations['high_priority'].append({
                    'category': '正の歪み対応',
                    'reason': f'歪度={skew:.3f} → 右裾が長い分布',
                    'features': ['sqrt_y', 'log_y', '高呼量フラグ(is_high_volume)', 'percentile_rank_rolling']
                })
            else:
                self.recommendations['medium_priority'].append({
                    'category': '負の歪み対応',
                    'reason': f'歪度={skew:.3f} → 左裾が長い分布',
                    'features': ['squared_y', '低呼量フラグ(is_low_volume)']
                })
        
        # 尖度診断
        kurt = stats_dict['kurtosis']
        if kurt > 3:
            self.recommendations['medium_priority'].append({
                'category': '重い裾対応',
                'reason': f'尖度={kurt:.3f} → 外れ値頻出',
                'features': ['Winsorization', '外れ値フラグ', 'レジーム分類(高/中/低)', 'Mahalanobis距離']
            })
    
    def _analyze_trend(self):
        """トレンド分析"""
        y = self.df[self.target_col].values
        t = np.arange(len(y))
        
        # 線形回帰
        model = LinearRegression()
        model.fit(t.reshape(-1, 1), y)
        trend = model.predict(t.reshape(-1, 1))
        r2 = model.score(t.reshape(-1, 1), y)
        slope = model.coef_[0]
        
        self.analysis_results['trend'] = {
            'r2': r2,
            'slope': slope,
            'trend_values': trend
        }
        
        # トレンド診断
        if r2 > 0.7:
            if abs(slope) > 0.01 * y.mean():
                trend_type = "強い線形トレンド" if slope > 0 else "強い減少トレンド"
                self.recommendations['essential'].append({
                    'category': 'トレンド特徴',
                    'reason': f'{trend_type} (R²={r2:.3f}, 傾き={slope:.2f})',
                    'features': ['t (時間インデックス)', 't_squared', 't_cubed', 'detrended = y - trend', 'growth_rate']
                })
        elif r2 > 0.3:
            self.recommendations['high_priority'].append({
                'category': '非線形トレンド',
                'reason': f'中程度の非線形トレンド (R²={r2:.3f})',
                'features': ['Spline特徴 (B-spline)', 'ローリング回帰係数', '区分線形特徴']
            })
        else:
            self.recommendations['optional'].append({
                'category': 'トレンド',
                'reason': f'トレンド弱い (R²={r2:.3f}) → トレンド特徴不要',
                'features': []
            })
    
    def _analyze_periodicity(self):
        """周期性分析"""
        df = self.df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # 曜日効果 (ANOVA)
        groups_dow = [df[df['dayofweek'] == i][self.target_col].values for i in range(7)]
        f_dow, p_dow = stats.f_oneway(*groups_dow)
        
        # 月効果
        groups_month = [df[df['month'] == i][self.target_col].values for i in range(1, 13)]
        f_month, p_month = stats.f_oneway(*groups_month)
        
        # 平日/週末
        df['is_weekend'] = df['dayofweek'] >= 5
        weekday_mean = df[~df['is_weekend']][self.target_col].mean()
        weekend_mean = df[df['is_weekend']][self.target_col].mean()
        t_stat, p_weekend = stats.ttest_ind(
            df[~df['is_weekend']][self.target_col],
            df[df['is_weekend']][self.target_col]
        )
        
        self.analysis_results['periodicity'] = {
            'dow_f': f_dow,
            'dow_p': p_dow,
            'month_f': f_month,
            'month_p': p_month,
            'weekend_p': p_weekend,
            'weekday_mean': weekday_mean,
            'weekend_mean': weekend_mean
        }
        
        # 曜日効果診断
        if p_dow < 0.001:
            self.recommendations['essential'].append({
                'category': '強い曜日効果',
                'reason': f'曜日ANOVA F値={f_dow:.2f}, p<0.001',
                'features': [
                    '曜日ワンホット (dow_mon, ..., dow_sun)',
                    '曜日循環エンコ (dow_sin, dow_cos)',
                    '曜日別統計 (dow_mean_*, dow_std_*)',
                    '過去4週同曜日平均',
                    '曜日×月交互作用'
                ]
            })
        elif p_weekend < 0.05:
            self.recommendations['high_priority'].append({
                'category': '平日/週末効果',
                'reason': f'平日={weekday_mean:.1f}, 週末={weekend_mean:.1f}, p={p_weekend:.4f}',
                'features': ['is_weekend', 'is_friday', 'is_monday']
            })
        
        # 月効果診断
        if p_month < 0.01:
            # 季節指数CV
            monthly_mean = df.groupby('month')[self.target_col].mean()
            seasonal_cv = monthly_mean.std() / monthly_mean.mean()
            
            self.recommendations['essential'].append({
                'category': '強い月次季節性',
                'reason': f'月ANOVA F値={f_month:.2f}, p<0.01, 季節CV={seasonal_cv:.3f}',
                'features': [
                    '月ワンホット (month_1, ..., month_12)',
                    '月循環エンコ (month_sin, month_cos)',
                    '四半期ダミー (q1, q2, q3, q4)',
                    'STL季節成分',
                    '季節調整済み値 (y / seasonal_index)'
                ]
            })
    
    def _analyze_autocorrelation(self):
        """自己相関分析"""
        y = self.df[self.target_col].dropna()
        
        # ACF/PACF計算 (最大60ラグ)
        max_lag = min(60, len(y) // 2)
        acf_values = acf(y, nlags=max_lag, fft=True)
        pacf_values = pacf(y, nlags=max_lag, method='ywm')
        
        # 有意性判定 (95%信頼区間)
        conf_interval = 1.96 / np.sqrt(len(y))
        
        significant_lags_acf = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
        significant_lags_pacf = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1
        
        self.analysis_results['autocorrelation'] = {
            'acf': acf_values,
            'pacf': pacf_values,
            'significant_lags_acf': significant_lags_acf.tolist(),
            'significant_lags_pacf': significant_lags_pacf.tolist()
        }
        
        # パターン診断
        acf_sig = significant_lags_acf[:10]  # 最初の10ラグ
        pacf_sig = significant_lags_pacf[:10]
        
        # AR(1)パターン
        if len(pacf_sig) > 0 and pacf_sig[0] == 1 and len(pacf_sig) < 3:
            self.recommendations['essential'].append({
                'category': 'AR(1)過程',
                'reason': 'PACF lag-1で切断 → 1次自己回帰',
                'features': ['lag_1 (最重要)', 'lag_1_squared', 'lag_1_cubed']
            })
        
        # 週次季節性
        if 7 in acf_sig or 7 in pacf_sig:
            self.recommendations['essential'].append({
                'category': '週次季節性',
                'reason': 'ACF/PACF lag-7有意 → 週次パターン',
                'features': [
                    'lag_7, lag_14, lag_21 (週次ラグ)',
                    'seasonal_diff_7 = y - lag_7',
                    'lag_7 × 曜日交互作用',
                    '同曜日ローリング平均'
                ]
            })
        
        # 月次季節性
        if 30 in acf_sig or 30 in pacf_sig:
            self.recommendations['high_priority'].append({
                'category': '月次季節性',
                'reason': 'ACF/PACF lag-30有意',
                'features': ['lag_30', 'seasonal_diff_30', 'monthly_rolling_mean']
            })
        
        # 複数有意ラグ
        if len(acf_sig) > 5:
            top_lags = acf_sig[:5]
            self.recommendations['high_priority'].append({
                'category': '複合自己回帰',
                'reason': f'複数ラグ有意: {top_lags.tolist()}',
                'features': [f'lag_{lag}' for lag in top_lags]
            })
    
    def _analyze_spectrum(self):
        """スペクトル解析"""
        y = self.df[self.target_col].fillna(method='ffill').values
        
        # FFT
        fft_values = fft(y)
        frequencies = fftfreq(len(y), d=1)  # 日次データ
        
        # パワースペクトル (正の周波数のみ)
        positive_freq_mask = frequencies > 0
        power = np.abs(fft_values[positive_freq_mask]) ** 2
        freq_positive = frequencies[positive_freq_mask]
        
        # 上位5周期
        top_k = 5
        top_indices = np.argsort(power)[-top_k:][::-1]
        top_periods = 1 / freq_positive[top_indices]
        top_power = power[top_indices]
        
        # 総パワーに対する割合
        total_power = power.sum()
        top_power_ratio = top_power / total_power
        
        self.analysis_results['spectrum'] = {
            'top_periods': top_periods,
            'top_power': top_power,
            'top_power_ratio': top_power_ratio
        }
        
        # 周期診断
        dominant_periods = []
        for period, ratio in zip(top_periods, top_power_ratio):
            if ratio > 0.05:  # 5%以上のパワー
                dominant_periods.append(period)
        
        if len(dominant_periods) > 0:
            period_str = ', '.join([f'{p:.1f}日' for p in dominant_periods[:3]])
            self.recommendations['high_priority'].append({
                'category': 'フーリエ特徴',
                'reason': f'主要周期検出: {period_str}',
                'features': [
                    f'各周期のsin/cos特徴: sin(2πt/{p:.1f}), cos(2πt/{p:.1f})'
                    for p in dominant_periods[:3]
                ] + ['高調波 (2倍周波数)']
            })
    
    def _analyze_stationarity(self):
        """定常性検定"""
        y = self.df[self.target_col].dropna()
        
        # ADF検定
        adf_result = adfuller(y, autolag='AIC')
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        
        # KPSS検定
        kpss_result = kpss(y, regression='ct', nlags='auto')
        kpss_statistic = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        
        self.analysis_results['stationarity'] = {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue
        }
        
        # 定常性診断
        if adf_pvalue > 0.05:
            self.recommendations['essential'].append({
                'category': '非定常性対応',
                'reason': f'ADF p値={adf_pvalue:.4f} > 0.05 → 非定常',
                'features': [
                    'diff_1 = y - lag_1 (1次差分)',
                    'diff_2 (2次差分)',
                    'log_return = log(y / lag_1)',
                    'detrended = y - linear_trend'
                ]
            })
        
        if kpss_pvalue < 0.05 and adf_pvalue < 0.05:
            self.recommendations['high_priority'].append({
                'category': 'トレンド定常',
                'reason': 'ADF定常 but KPSS非定常 → トレンド定常',
                'features': ['トレンド除去のみ (差分不要)']
            })
    
    def _analyze_seasonal_decomposition(self):
        """季節分解"""
        try:
            stl = STL(self.df[self.target_col], seasonal=7, robust=True)
            result = stl.fit()
            
            seasonal = result.seasonal
            trend = result.trend
            resid = result.resid
            
            # 成分の分散比
            var_total = self.df[self.target_col].var()
            var_seasonal = seasonal.var()
            var_trend = trend.var()
            var_resid = resid.var()
            
            seasonal_strength = 1 - var_resid / (var_seasonal + var_resid)
            trend_strength = 1 - var_resid / (var_trend + var_resid)
            
            self.analysis_results['decomposition'] = {
                'seasonal': seasonal,
                'trend': trend,
                'resid': resid,
                'seasonal_strength': seasonal_strength,
                'trend_strength': trend_strength
            }
            
            # 季節成分診断
            if seasonal_strength > 0.6:
                self.recommendations['essential'].append({
                    'category': '強い季節成分',
                    'reason': f'季節強度={seasonal_strength:.3f} > 0.6',
                    'features': [
                        'seasonal_component (STLから抽出)',
                        'seasonally_adjusted = y - seasonal',
                        'seasonal_ratio = seasonal / trend'
                    ]
                })
            
            # トレンド成分診断
            if trend_strength > 0.6:
                self.recommendations['high_priority'].append({
                    'category': '強いトレンド成分',
                    'reason': f'トレンド強度={trend_strength:.3f} > 0.6',
                    'features': [
                        'trend_component',
                        'detrended = y - trend',
                        'trend_pct_change'
                    ]
                })
            
            # 残差診断
            if var_resid / var_total > 0.3:
                self.recommendations['medium_priority'].append({
                    'category': '大きな残差',
                    'reason': f'残差分散比={var_resid/var_total:.3f} > 0.3 → 外生変数必要',
                    'features': [
                        '外生変数追加 (プロモーション、天候等)',
                        '残差ベース異常検出',
                        'residual_lag_1'
                    ]
                })
        except:
            self.analysis_results['decomposition'] = None
    
    def _analyze_outliers(self):
        """異常値分析"""
        y = self.df[self.target_col]
        
        # IQR法
        Q1, Q3 = y.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_iqr = ((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR))
        
        # Zスコア法
        z_scores = np.abs((y - y.mean()) / y.std())
        outlier_zscore = z_scores > 3
        
        outlier_rate = outlier_iqr.sum() / len(y)
        
        self.analysis_results['outliers'] = {
            'outlier_rate': outlier_rate,
            'outlier_count': outlier_iqr.sum()
        }
        
        # 異常値診断
        if outlier_rate > 0.05:
            self.recommendations['high_priority'].append({
                'category': '頻出異常値対応',
                'reason': f'異常値率={outlier_rate:.2%} > 5%',
                'features': [
                    'Winsorization (y_winsorized)',
                    '外れ値フラグ (is_outlier_iqr, is_outlier_zscore)',
                    'days_since_last_outlier',
                    'レジーム分類 (高/中/低)'
                ]
            })
        elif outlier_rate > 0.01:
            self.recommendations['medium_priority'].append({
                'category': '散発的異常値',
                'reason': f'異常値率={outlier_rate:.2%}',
                'features': ['外れ値フラグ', 'Winsorization']
            })
    
    def _analyze_calendar_effects(self):
        """カレンダー効果分析"""
        df = self.df.copy()
        df['day_of_month'] = df.index.day
        df['days_in_month'] = df.index.days_in_month
        
        # 月初月末効果
        df['is_month_start'] = df['day_of_month'] <= 5
        df['is_month_end'] = df['day_of_month'] >= df['days_in_month'] - 5
        df['is_mid_month'] = (df['day_of_month'] >= 10) & (df['day_of_month'] <= 20)
        
        month_start_mean = df[df['is_month_start']][self.target_col].mean()
        month_mid_mean = df[df['is_mid_month']][self.target_col].mean()
        month_end_mean = df[df['is_month_end']][self.target_col].mean()
        
        # t検定
        t_start, p_start = stats.ttest_ind(
            df[df['is_month_start']][self.target_col],
            df[df['is_mid_month']][self.target_col]
        )
        
        self.analysis_results['calendar'] = {
            'month_start_mean': month_start_mean,
            'month_mid_mean': month_mid_mean,
            'month_end_mean': month_end_mean,
            'p_start': p_start
        }
        
        # カレンダー効果診断
        if p_start < 0.05:
            self.recommendations['high_priority'].append({
                'category': '月初月末効果',
                'reason': f'月初平均={month_start_mean:.1f}, 月中={month_mid_mean:.1f}, p={p_start:.4f}',
                'features': [
                    'is_month_start (1-5日)',
                    'is_month_end (26-31日)',
                    'days_to_month_end',
                    'is_payday_week (給与サイクル)',
                    'week_of_month'
                ]
            })
    
    def _analyze_volatility(self):
        """変動性分析"""
        # ローリング標準偏差
        rolling_std_30 = self.df[self.target_col].rolling(30).std()
        
        # 曜日別変動
        df = self.df.copy()
        df['dayofweek'] = df.index.dayofweek
        dow_std = df.groupby('dayofweek')[self.target_col].std()
        dow_cv = dow_std / df.groupby('dayofweek')[self.target_col].mean()
        
        max_cv_diff = dow_cv.max() - dow_cv.min()
        
        self.analysis_results['volatility'] = {
            'rolling_std_30': rolling_std_30,
            'dow_cv': dow_cv,
            'max_cv_diff': max_cv_diff
        }
        
        # 不均一分散診断
        if max_cv_diff > 0.1:
            self.recommendations['medium_priority'].append({
                'category': '曜日別変動差',
                'reason': f'曜日別CVの差={max_cv_diff:.3f} > 0.1',
                'features': [
                    'dow_std_historical (曜日別標準偏差)',
                    'rolling_var_30',
                    'volatility_regime (高/低ボラティリティ)'
                ]
            })
        
        # 時変変動性
        if rolling_std_30.std() / rolling_std_30.mean() > 0.3:
            self.recommendations['high_priority'].append({
                'category': '時変ボラティリティ',
                'reason': 'ローリング標準偏差が時変',
                'features': [
                    'squared_residual_lag_1 (GARCH型)',
                    'realized_vol_7, realized_vol_30',
                    'abs_residual_lag_1'
                ]
            })
    
    def _analyze_lag_importance(self):
        """ラグ重要度分析 (簡易相関ベース)"""
        y = self.df[self.target_col]
        
        # 主要ラグとの相関
        important_lags = [1, 2, 3, 7, 14, 21, 30, 60]
        lag_correlations = {}
        
        for lag in important_lags:
            if lag < len(y):
                corr = y.corr(y.shift(lag))
                if not np.isnan(corr):
                    lag_correlations[lag] = abs(corr)
        
        # 相関上位ラグ
        sorted_lags = sorted(lag_correlations.items(), key=lambda x: x[1], reverse=True)
        top_lags = [lag for lag, corr in sorted_lags if corr > 0.3][:5]
        
        self.analysis_results['lag_importance'] = lag_correlations
        
        if len(top_lags) > 0:
            self.recommendations['high_priority'].append({
                'category': '重要ラグ特徴',
                'reason': f'高相関ラグ: {top_lags}',
                'features': [f'lag_{lag} (相関={lag_correlations[lag]:.3f})' for lag in top_lags]
            })
    
    def _generate_recommendations(self):
        """推奨特徴量の優先順位整理"""
        # 重複排除とカテゴリ整理
        pass  # すでに各分析で追加済み
    
    def _save_detailed_report(self, output_dir):
        """詳細レポート保存"""
        report_path = f"{output_dir}/feature_recommendation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("コールセンター呼量予測 - 特徴量推奨レポート\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析日時: {pd.Timestamp.now()}\n")
            f.write(f"データ期間: {self.df.index.min()} ~ {self.df.index.max()}\n")
            f.write(f"データ件数: {len(self.df)} 日\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("【分析結果サマリー】\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本統計量
            stats = self.analysis_results['basic_stats']
            f.write("■ 基本統計量\n")
            f.write(f"  平均: {stats['mean']:.2f}\n")
            f.write(f"  標準偏差: {stats['std']:.2f}\n")
            f.write(f"  変動係数 (CV): {stats['cv']:.4f}\n")
            f.write(f"  歪度: {stats['skewness']:.4f}\n")
            f.write(f"  尖度: {stats['kurtosis']:.4f}\n\n")
            
            # トレンド
            trend = self.analysis_results['trend']
            f.write("■ トレンド分析\n")
            f.write(f"  R²: {trend['r2']:.4f}\n")
            f.write(f"  傾き: {trend['slope']:.4f}\n\n")
            
            # 定常性
            stationarity = self.analysis_results['stationarity']
            f.write("■ 定常性検定\n")
            f.write(f"  ADF p値: {stationarity['adf_pvalue']:.4f} ({'定常' if stationarity['adf_pvalue'] < 0.05 else '非定常'})\n")
            f.write(f"  KPSS p値: {stationarity['kpss_pvalue']:.4f} ({'定常' if stationarity['kpss_pvalue'] > 0.05 else '非定常'})\n\n")
            
            # 周期性
            periodicity = self.analysis_results['periodicity']
            f.write("■ 周期性分析\n")
            f.write(f"  曜日効果 F値: {periodicity['dow_f']:.2f}, p値: {periodicity['dow_p']:.4e}\n")
            f.write(f"  月効果 F値: {periodicity['month_f']:.2f}, p値: {periodicity['month_p']:.4e}\n")
            f.write(f"  平日/週末 p値: {periodicity['weekend_p']:.4f}\n\n")
            
            # 自己相関
            autocorr = self.analysis_results['autocorrelation']
            f.write("■ 自己相関\n")
            f.write(f"  有意なACFラグ (上位10): {autocorr['significant_lags_acf'][:10]}\n")
            f.write(f"  有意なPACFラグ (上位10): {autocorr['significant_lags_pacf'][:10]}\n\n")
            
            # スペクトル
            spectrum = self.analysis_results['spectrum']
            f.write("■ スペクトル解析\n")
            f.write(f"  主要周期 (上位3): {[f'{p:.1f}日' for p in spectrum['top_periods'][:3]]}\n\n")
            
            # 異常値
            outliers = self.analysis_results['outliers']
            f.write("■ 異常値\n")
            f.write(f"  異常値率: {outliers['outlier_rate']:.2%} ({outliers['outlier_count']}件)\n\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("【推奨特徴量一覧】\n")
            f.write("=" * 80 + "\n\n")
            
            # 必須特徴量
            f.write("■ 必須特徴量 (Essential)\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(self.recommendations['essential'], 1):
                f.write(f"\n[{i}] {rec['category']}\n")
                f.write(f"    理由: {rec['reason']}\n")
                f.write(f"    特徴量:\n")
                for feat in rec['features']:
                    f.write(f"      - {feat}\n")
            
            # 高優先度
            f.write("\n\n■ 高優先度特徴量 (High Priority)\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(self.recommendations['high_priority'], 1):
                f.write(f"\n[{i}] {rec['category']}\n")
                f.write(f"    理由: {rec['reason']}\n")
                f.write(f"    特徴量:\n")
                for feat in rec['features']:
                    f.write(f"      - {feat}\n")
            
            # 中優先度
            f.write("\n\n■ 中優先度特徴量 (Medium Priority)\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(self.recommendations['medium_priority'], 1):
                f.write(f"\n[{i}] {rec['category']}\n")
                f.write(f"    理由: {rec['reason']}\n")
                f.write(f"    特徴量:\n")
                for feat in rec['features']:
                    f.write(f"      - {feat}\n")
            
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("レポート終了\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ 詳細レポート保存: {report_path}")
    
    def _save_feature_generation_code(self, output_dir):
        """特徴量生成コード保存"""
        code_path = f"{output_dir}/generate_features.py"
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('自動生成された特徴量生成コード\n')
            f.write('分析結果に基づく推奨特徴量を実装\n')
            f.write('"""\n\n')
            
            f.write('import pandas as pd\n')
            f.write('import numpy as np\n')
            f.write('from scipy.stats import boxcox\n')
            f.write('from statsmodels.tsa.seasonal import STL\n')
            f.write('import jpholiday\n\n')
            
            f.write('def generate_recommended_features(df, date_col="ds", target_col="y"):\n')
            f.write('    """\n')
            f.write('    推奨特徴量を生成\n')
            f.write('    \n')
            f.write('    Parameters:\n')
            f.write('    -----------\n')
            f.write('    df : pd.DataFrame\n')
            f.write('        入力データ (ds, y 形式)\n')
            f.write('    \n')
            f.write('    Returns:\n')
            f.write('    --------\n')
            f.write('    pd.DataFrame\n')
            f.write('        特徴量追加後のデータ\n')
            f.write('    """\n')
            f.write('    df = df.copy()\n')
            f.write('    df[date_col] = pd.to_datetime(df[date_col])\n')
            f.write('    df = df.set_index(date_col).sort_index()\n\n')
            
            f.write('    # 基本時間特徴\n')
            f.write('    df["t"] = np.arange(len(df))\n')
            f.write('    df["dayofweek"] = df.index.dayofweek\n')
            f.write('    df["month"] = df.index.month\n')
            f.write('    df["quarter"] = df.index.quarter\n')
            f.write('    df["day_of_month"] = df.index.day\n')
            f.write('    df["week_of_year"] = df.index.isocalendar().week\n')
            f.write('    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)\n\n')
            
            # 必須特徴量からコード生成
            all_recs = (self.recommendations['essential'] + 
                       self.recommendations['high_priority'])
            
            for rec in all_recs:
                category = rec['category']
                
                if '曜日' in category or 'dow' in category.lower():
                    f.write('    # 曜日特徴\n')
                    f.write('    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)\n')
                    f.write('    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)\n')
                    f.write('    for dow in range(7):\n')
                    f.write('        df[f"dow_{dow}"] = (df["dayofweek"] == dow).astype(int)\n\n')
                
                if 'ラグ' in category or 'lag' in category.lower():
                    f.write('    # ラグ特徴\n')
                    f.write('    for lag in [1, 2, 7, 14, 21, 30]:\n')
                    f.write('        df[f"lag_{lag}"] = df[target_col].shift(lag)\n\n')
                
                if 'ローリング' in category or 'rolling' in category.lower():
                    f.write('    # ローリング統計\n')
                    f.write('    for window in [7, 14, 30]:\n')
                    f.write('        df[f"rolling_mean_{window}"] = df[target_col].rolling(window).mean()\n')
                    f.write('        df[f"rolling_std_{window}"] = df[target_col].rolling(window).std()\n')
                    f.write('        df[f"ewm_{window}"] = df[target_col].ewm(span=window).mean()\n\n')
                
                if 'トレンド' in category or 'trend' in category.lower():
                    f.write('    # トレンド特徴\n')
                    f.write('    df["t_squared"] = df["t"] ** 2\n\n')
                
                if '月' in category and '曜日' not in category:
                    f.write('    # 月特徴\n')
                    f.write('    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)\n')
                    f.write('    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)\n')
                    f.write('    for m in range(1, 13):\n')
                    f.write('        df[f"month_{m}"] = (df["month"] == m).astype(int)\n\n')
                
                if '差分' in category or 'diff' in category.lower():
                    f.write('    # 差分特徴\n')
                    f.write('    df["diff_1"] = df[target_col].diff()\n')
                    f.write('    df["diff_7"] = df[target_col].diff(7)\n\n')
                
                if 'GARCH' in category or 'ボラティリティ' in category:
                    f.write('    # ボラティリティ特徴\n')
                    f.write('    df["rolling_std_30"] = df[target_col].rolling(30).std()\n')
                    f.write('    df["realized_vol_7"] = df[target_col].rolling(7).std() * np.sqrt(7)\n\n')
                
                if '月初月末' in category:
                    f.write('    # カレンダー特徴\n')
                    f.write('    df["is_month_start"] = (df.index.day <= 5).astype(int)\n')
                    f.write('    df["is_month_end"] = (df.index.day >= df.index.days_in_month - 5).astype(int)\n\n')
            
            f.write('    # 祝日特徴\n')
            f.write('    df["is_holiday"] = df.index.to_series().apply(lambda x: jpholiday.is_holiday(x)).astype(int)\n\n')
            
            f.write('    return df\n\n')
            
            f.write('# 使用例\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    df = pd.read_csv("your_data.csv")  # ds, y カラム\n')
            f.write('    df_featured = generate_recommended_features(df)\n')
            f.write('    df_featured = df_featured.dropna()  # 欠損値除去\n')
            f.write('    print(f"Generated {len(df_featured.columns)} features")\n')
            f.write('    print(df_featured.head())\n')
        
        print(f"✓ 特徴量生成コード保存: {code_path}")
    
    def _create_visualization_dashboard(self, output_dir):
        """可視化ダッシュボード作成"""
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('特徴量推奨システム - 分析ダッシュボード', fontsize=16, fontweight='bold')
        
        y = self.df[self.target_col]
        
        # 1. 時系列プロット
        axes[0, 0].plot(self.df.index, y, linewidth=0.8)
        axes[0, 0].set_title('時系列プロット')
        axes[0, 0].set_xlabel('日付')
        axes[0, 0].set_ylabel(self.target_col)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ヒストグラム + 統計量
        axes[0, 1].hist(y, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('分布')
        stats = self.analysis_results['basic_stats']
        axes[0, 1].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean={stats["mean"]:.1f}')
        axes[0, 1].legend()
        textstr = f'CV={stats["cv"]:.3f}\nSkew={stats["skewness"]:.3f}\nKurt={stats["kurtosis"]:.3f}'
        axes[0, 1].text(0.7, 0.95, textstr, transform=axes[0, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 曜日別
        df_dow = self.df.copy()
        df_dow['dayofweek'] = df_dow.index.dayofweek
        dow_mean = df_dow.groupby('dayofweek')[self.target_col].mean()
        axes[0, 2].bar(range(7), dow_mean, color='steelblue', alpha=0.7)
        axes[0, 2].set_title('曜日別平均')
        axes[0, 2].set_xlabel('曜日 (0=月)')
        axes[0, 2].set_xticks(range(7))
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. ACF
        acf_vals = self.analysis_results['autocorrelation']['acf']
        axes[1, 0].stem(range(len(acf_vals)), acf_vals, basefmt=' ', use_line_collection=True)
        axes[1, 0].axhline(1.96/np.sqrt(len(y)), color='red', linestyle='--', linewidth=0.8)
        axes[1, 0].axhline(-1.96/np.sqrt(len(y)), color='red', linestyle='--', linewidth=0.8)
        axes[1, 0].set_title('ACF')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. PACF
        pacf_vals = self.analysis_results['autocorrelation']['pacf']
        axes[1, 1].stem(range(len(pacf_vals)), pacf_vals, basefmt=' ', use_line_collection=True)
        axes[1, 1].axhline(1.96/np.sqrt(len(y)), color='red', linestyle='--', linewidth=0.8)
        axes[1, 1].axhline(-1.96/np.sqrt(len(y)), color='red', linestyle='--', linewidth=0.8)
        axes[1, 1].set_title('PACF')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 月別
        df_month = self.df.copy()
        df_month['month'] = df_month.index.month
        month_mean = df_month.groupby('month')[self.target_col].mean()
        axes[1, 2].bar(range(1, 13), month_mean, color='coral', alpha=0.7)
        axes[1, 2].set_title('月別平均')
        axes[1, 2].set_xlabel('月')
        axes[1, 2].set_xticks(range(1, 13))
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. トレンド
        trend_vals = self.analysis_results['trend']['trend_values']
        axes[2, 0].plot(self.df.index, y, alpha=0.5, label='Original')
        axes[2, 0].plot(self.df.index, trend_vals, color='red', linewidth=2, label='Trend')
        axes[2, 0].set_title(f'トレンド (R²={self.analysis_results["trend"]["r2"]:.3f})')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. STL分解 (季節成分)
        if self.analysis_results['decomposition'] is not None:
            seasonal = self.analysis_results['decomposition']['seasonal']
            axes[2, 1].plot(seasonal.index[:365], seasonal.values[:365])  # 最初の1年
            axes[2, 1].set_title('季節成分 (最初の1年)')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'STL分解失敗', ha='center', va='center')
        
        # 9. ローリング標準偏差
        rolling_std = y.rolling(30).std()
        axes[2, 2].plot(self.df.index, rolling_std, color='purple')
        axes[2, 2].set_title('ローリング標準偏差 (30日)')
        axes[2, 2].set_ylabel('Std')
        axes[2, 2].grid(True, alpha=0.3)
        
        # 10. ラグプロット (lag=1)
        axes[3, 0].scatter(y.shift(1), y, alpha=0.3, s=5)
        axes[3, 0].set_title('ラグプロット (lag=1)')
        axes[3, 0].set_xlabel('y(t-1)')
        axes[3, 0].set_ylabel('y(t)')
        axes[3, 0].grid(True, alpha=0.3)
        
        # 11. ラグプロット (lag=7)
        axes[3, 1].scatter(y.shift(7), y, alpha=0.3, s=5, color='orange')
        axes[3, 1].set_title('ラグプロット (lag=7)')
        axes[3, 1].set_xlabel('y(t-7)')
        axes[3, 1].set_ylabel('y(t)')
        axes[3, 1].grid(True, alpha=0.3)
        
        # 12. 推奨特徴量サマリー
        axes[3, 2].axis('off')
        summary_text = "【推奨特徴量サマリー】\n\n"
        summary_text += f"必須: {len(self.recommendations['essential'])}カテゴリ\n"
        summary_text += f"高優先度: {len(self.recommendations['high_priority'])}カテゴリ\n"
        summary_text += f"中優先度: {len(self.recommendations['medium_priority'])}カテゴリ\n\n"
        summary_text += "詳細はレポートを参照"
        axes[3, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        dashboard_path = f"{output_dir}/analysis_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ダッシュボード保存: {dashboard_path}")


def main():
    """メイン実行関数"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python feature_recommendation_system.py <csv_file>")
        print("Example: python feature_recommendation_system.py call_center_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # データ読み込み
    print(f"データ読み込み: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # システム初期化
    system = FeatureRecommendationSystem(df)
    
    # 完全分析実行
    recommendations = system.run_complete_analysis(output_dir='./feature_recommendations')
    
    print("\n" + "="*80)
    print("推奨特徴量の概要:")
    print("="*80)
    
    print("\n【必須特徴量】")
    for rec in recommendations['essential']:
        print(f"  - {rec['category']}: {rec['reason']}")
    
    print("\n【高優先度特徴量】")
    for rec in recommendations['high_priority']:
        print(f"  - {rec['category']}: {rec['reason']}")
    
    print("\n【中優先度特徴量】")
    for rec in recommendations['medium_priority']:
        print(f"  - {rec['category']}: {rec['reason']}")


if __name__ == "__main__":
    main()
