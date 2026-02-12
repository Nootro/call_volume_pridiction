"""
コールセンター呼量予測のための包括的EDA分析スクリプト
Prophet形式データ（ds, y）に対応
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CallCenterEDA:
    """コールセンターデータの包括的EDA分析クラス"""
    
    def __init__(self, df):
        """
        Parameters:
        -----------
        df : DataFrame
            'ds' (日付) と 'y' (呼量) のカラムを持つDataFrame
        """
        self.df = df.copy()
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df = self.df.sort_values('ds').reset_index(drop=True)
        self.df.set_index('ds', inplace=True)
        
        # 追加特徴量の作成
        self._create_features()
        
    def _create_features(self):
        """時系列特徴量の作成"""
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['quarter'] = self.df.index.quarter
        self.df['dayofyear'] = self.df.index.dayofyear
        self.df['weekofyear'] = self.df.index.isocalendar().week
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        self.df['is_month_start'] = self.df.index.is_month_start.astype(int)
        self.df['is_month_end'] = self.df.index.is_month_end.astype(int)
        
    def run_all_analyses(self, output_dir='./eda_output'):
        """全ての分析を実行"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("コールセンター呼量予測 - 包括的EDA分析")
        print("=" * 80)
        
        # 1. 基本統計量
        self.basic_statistics()
        
        # 2. 時系列プロット
        self.plot_timeseries(output_dir)
        
        # 3. 分布分析
        self.plot_distribution(output_dir)
        
        # 4. 移動平均分析
        self.plot_moving_averages(output_dir)
        
        # 5. 周期性分析
        self.plot_periodicity(output_dir)
        
        # 6. 自己相関・偏自己相関分析
        self.plot_acf_pacf(output_dir)
        
        # 7. スペクトル解析
        self.spectral_analysis(output_dir)
        
        # 8. 定常性検定
        self.stationarity_tests()
        
        # 9. 季節分解
        self.seasonal_decomposition(output_dir)
        
        # 10. トレンド分析
        self.trend_analysis(output_dir)
        
        # 11. 異常値検出
        self.outlier_detection(output_dir)
        
        # 12. 曜日・月別パターン
        self.calendar_patterns(output_dir)
        
        # 13. 変動係数分析
        self.variability_analysis(output_dir)
        
        # 14. ヒートマップ分析
        self.heatmap_analysis(output_dir)
        
        # 15. ラグプロット
        self.lag_plots(output_dir)
        
        # 16. 差分分析
        self.difference_analysis(output_dir)
        
        print("\n" + "=" * 80)
        print(f"全ての分析が完了しました。結果は '{output_dir}' に保存されています。")
        print("=" * 80)
        
    def basic_statistics(self):
        """基本統計量の表示"""
        print("\n【1. 基本統計量】")
        print("-" * 80)
        print(f"データ期間: {self.df.index.min().date()} ~ {self.df.index.max().date()}")
        print(f"データ数: {len(self.df)} 日")
        print(f"\n記述統計量:")
        print(self.df['y'].describe())
        print(f"\n変動係数 (CV): {self.df['y'].std() / self.df['y'].mean():.4f}")
        print(f"歪度 (Skewness): {self.df['y'].skew():.4f}")
        print(f"尖度 (Kurtosis): {self.df['y'].kurtosis():.4f}")
        
    def plot_timeseries(self, output_dir):
        """時系列プロット"""
        print("\n【2. 時系列プロット】")
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 全期間
        axes[0].plot(self.df.index, self.df['y'], linewidth=0.8, alpha=0.8)
        axes[0].set_title('Call Volume Time Series (Full Period)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Call Volume')
        axes[0].grid(True, alpha=0.3)
        
        # 年別
        for year in self.df['year'].unique():
            year_data = self.df[self.df['year'] == year]
            axes[1].plot(year_data.index, year_data['y'], label=f'{year}', alpha=0.7)
        axes[1].set_title('Call Volume by Year', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Call Volume')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 最近90日
        recent_data = self.df.tail(90)
        axes[2].plot(recent_data.index, recent_data['y'], linewidth=1.5, color='darkblue')
        axes[2].set_title('Call Volume - Recent 90 Days', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Call Volume')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 01_timeseries.png")
        
    def plot_distribution(self, output_dir):
        """分布分析"""
        print("\n【3. 分布分析】")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ヒストグラム
        axes[0, 0].hist(self.df['y'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df['y'].mean(), color='red', linestyle='--', label=f"Mean: {self.df['y'].mean():.2f}")
        axes[0, 0].axvline(self.df['y'].median(), color='green', linestyle='--', label=f"Median: {self.df['y'].median():.2f}")
        axes[0, 0].set_title('Distribution of Call Volume', fontweight='bold')
        axes[0, 0].set_xlabel('Call Volume')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Qプロット
        stats.probplot(self.df['y'], dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(self.df['y'], vert=True)
        axes[1, 0].set_title('Box Plot of Call Volume', fontweight='bold')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].grid(True, alpha=0.3)
        
        # KDE plot
        self.df['y'].plot(kind='kde', ax=axes[1, 1], linewidth=2)
        axes[1, 1].set_title('Kernel Density Estimation', fontweight='bold')
        axes[1, 1].set_xlabel('Call Volume')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 02_distribution.png")
        
    def plot_moving_averages(self, output_dir):
        """移動平均分析"""
        print("\n【4. 移動平均分析】")
        
        windows = [7, 14, 30, 60]
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 移動平均
        axes[0].plot(self.df.index, self.df['y'], label='Original', alpha=0.5, linewidth=0.8)
        for window in windows:
            ma = self.df['y'].rolling(window=window).mean()
            axes[0].plot(self.df.index, ma, label=f'MA-{window}', linewidth=1.5)
        axes[0].set_title('Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Call Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 指数移動平均
        axes[1].plot(self.df.index, self.df['y'], label='Original', alpha=0.5, linewidth=0.8)
        for window in windows:
            ema = self.df['y'].ewm(span=window, adjust=False).mean()
            axes[1].plot(self.df.index, ema, label=f'EMA-{window}', linewidth=1.5)
        axes[1].set_title('Exponential Moving Averages', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Call Volume')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_moving_averages.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 03_moving_averages.png")
        
    def plot_periodicity(self, output_dir):
        """周期性分析"""
        print("\n【5. 周期性分析】")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 曜日別
        dayofweek_mean = self.df.groupby('dayofweek')['y'].mean()
        axes[0, 0].bar(range(7), dayofweek_mean.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xticks(range(7))
        axes[0, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 0].set_title('Average Call Volume by Day of Week', fontweight='bold')
        axes[0, 0].set_ylabel('Call Volume')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 月別
        month_mean = self.df.groupby('month')['y'].mean()
        axes[0, 1].bar(range(1, 13), month_mean.values, color='coral', alpha=0.7)
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_title('Average Call Volume by Month', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Call Volume')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 四半期別
        quarter_mean = self.df.groupby('quarter')['y'].mean()
        axes[1, 0].bar(range(1, 5), quarter_mean.values, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xticks(range(1, 5))
        axes[1, 0].set_title('Average Call Volume by Quarter', fontweight='bold')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 週末vs平日
        weekend_comparison = self.df.groupby('is_weekend')['y'].mean()
        axes[1, 1].bar(['Weekday', 'Weekend'], weekend_comparison.values, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        axes[1, 1].set_title('Weekday vs Weekend Call Volume', fontweight='bold')
        axes[1, 1].set_ylabel('Call Volume')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_periodicity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 04_periodicity.png")
        
    def plot_acf_pacf(self, output_dir):
        """自己相関・偏自己相関分析"""
        print("\n【6. 自己相関・偏自己相関分析】")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # ACF
        plot_acf(self.df['y'].dropna(), lags=60, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Lag (days)')
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(self.df['y'].dropna(), lags=60, ax=axes[1], alpha=0.05)
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Lag (days)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_acf_pacf.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 05_acf_pacf.png")
        
    def spectral_analysis(self, output_dir):
        """スペクトル解析（フーリエ変換）"""
        print("\n【7. スペクトル解析】")
        
        # 欠損値を補間
        y_interpolated = self.df['y'].interpolate()
        
        # FFT
        fft_values = np.fft.fft(y_interpolated)
        fft_freq = np.fft.fftfreq(len(y_interpolated))
        
        # パワースペクトル
        power = np.abs(fft_values) ** 2
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 周波数スペクトル
        positive_freq_idx = fft_freq > 0
        axes[0].plot(fft_freq[positive_freq_idx], power[positive_freq_idx])
        axes[0].set_title('Power Spectrum (Frequency Domain)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frequency')
        axes[0].set_ylabel('Power')
        axes[0].grid(True, alpha=0.3)
        
        # 周期（日数）に変換
        periods = 1 / fft_freq[positive_freq_idx]
        axes[1].plot(periods, power[positive_freq_idx])
        axes[1].set_xlim(0, 100)  # 0-100日の周期を表示
        axes[1].set_title('Power Spectrum (Period Domain)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Period (days)')
        axes[1].set_ylabel('Power')
        axes[1].grid(True, alpha=0.3)
        
        # 主要な周期を特定
        top_periods_idx = np.argsort(power[positive_freq_idx])[-5:][::-1]
        top_periods = periods[top_periods_idx]
        print(f"  主要な周期（日）: {', '.join([f'{p:.2f}' for p in top_periods if p < 365])}")
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 06_spectral_analysis.png")
        
    def stationarity_tests(self):
        """定常性検定"""
        print("\n【8. 定常性検定】")
        print("-" * 80)
        
        # ADF検定（拡張ディッキー・フラー検定）
        adf_result = adfuller(self.df['y'].dropna())
        print("ADF検定 (Augmented Dickey-Fuller Test):")
        print(f"  統計量: {adf_result[0]:.4f}")
        print(f"  p値: {adf_result[1]:.4f}")
        print(f"  判定: {'定常' if adf_result[1] < 0.05 else '非定常'} (p < 0.05で定常)")
        
        # KPSS検定
        kpss_result = kpss(self.df['y'].dropna(), regression='ct')
        print(f"\nKPSS検定:")
        print(f"  統計量: {kpss_result[0]:.4f}")
        print(f"  p値: {kpss_result[1]:.4f}")
        print(f"  判定: {'定常' if kpss_result[1] > 0.05 else '非定常'} (p > 0.05で定常)")
        
    def seasonal_decomposition(self, output_dir):
        """季節分解（STL分解）"""
        print("\n【9. 季節分解（STL分解）】")
        
        # 加法分解
        decomposition_add = seasonal_decompose(self.df['y'], model='additive', period=7)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition_add.observed.plot(ax=axes[0])
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Seasonal Decomposition (Additive Model)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        decomposition_add.trend.plot(ax=axes[1])
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        decomposition_add.seasonal.plot(ax=axes[2])
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        decomposition_add.resid.plot(ax=axes[3])
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 07_seasonal_decomposition.png")
        
    def trend_analysis(self, output_dir):
        """トレンド分析"""
        print("\n【10. トレンド分析】")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 年次トレンド
        yearly_data = self.df.groupby('year')['y'].agg(['mean', 'std', 'min', 'max'])
        x = yearly_data.index
        axes[0, 0].plot(x, yearly_data['mean'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].fill_between(x, yearly_data['mean'] - yearly_data['std'], 
                                yearly_data['mean'] + yearly_data['std'], alpha=0.3)
        axes[0, 0].set_title('Yearly Trend (Mean ± Std)', fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Call Volume')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 月次トレンド（全期間）
        monthly_data = self.df.resample('ME')['y'].mean()
        axes[0, 1].plot(monthly_data.index, monthly_data.values, linewidth=1.5)
        z = np.polyfit(range(len(monthly_data)), monthly_data.values, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(monthly_data.index, p(range(len(monthly_data))), 
                       "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[0, 1].set_title('Monthly Trend with Linear Fit', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Call Volume')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 四半期別トレンド
        quarterly_data = self.df.resample('QE')['y'].mean()
        axes[1, 0].bar(range(len(quarterly_data)), quarterly_data.values, alpha=0.7)
        axes[1, 0].set_title('Quarterly Average Call Volume', fontweight='bold')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 成長率
        monthly_pct_change = monthly_data.pct_change() * 100
        axes[1, 1].plot(monthly_pct_change.index, monthly_pct_change.values, linewidth=1.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Month-over-Month Growth Rate (%)', fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Growth Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 08_trend_analysis.png")
        
    def outlier_detection(self, output_dir):
        """異常値検出"""
        print("\n【11. 異常値検出】")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IQR法
        Q1 = self.df['y'].quantile(0.25)
        Q3 = self.df['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = (self.df['y'] < lower_bound) | (self.df['y'] > upper_bound)
        
        axes[0, 0].plot(self.df.index, self.df['y'], alpha=0.5, linewidth=0.8)
        axes[0, 0].scatter(self.df.index[outliers_iqr], self.df.loc[outliers_iqr, 'y'], 
                          color='red', s=30, label=f'Outliers (IQR): {outliers_iqr.sum()}')
        axes[0, 0].axhline(y=upper_bound, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=lower_bound, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Outlier Detection (IQR Method)', fontweight='bold')
        axes[0, 0].set_ylabel('Call Volume')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Z-score法
        z_scores = np.abs(stats.zscore(self.df['y']))
        outliers_zscore = z_scores > 3
        
        axes[0, 1].plot(self.df.index, self.df['y'], alpha=0.5, linewidth=0.8)
        axes[0, 1].scatter(self.df.index[outliers_zscore], self.df.loc[outliers_zscore, 'y'], 
                          color='red', s=30, label=f'Outliers (Z-score>3): {outliers_zscore.sum()}')
        axes[0, 1].set_title('Outlier Detection (Z-score Method)', fontweight='bold')
        axes[0, 1].set_ylabel('Call Volume')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 移動平均からの乖離
        ma_30 = self.df['y'].rolling(window=30).mean()
        deviation = np.abs(self.df['y'] - ma_30)
        outliers_ma = deviation > (deviation.std() * 2.5)
        
        axes[1, 0].plot(self.df.index, self.df['y'], alpha=0.5, linewidth=0.8, label='Original')
        axes[1, 0].plot(self.df.index, ma_30, color='green', linewidth=1.5, label='MA-30')
        axes[1, 0].scatter(self.df.index[outliers_ma], self.df.loc[outliers_ma, 'y'], 
                          color='red', s=30, label=f'Outliers: {outliers_ma.sum()}')
        axes[1, 0].set_title('Outlier Detection (Deviation from MA-30)', fontweight='bold')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 異常値の割合
        outlier_methods = ['IQR', 'Z-score', 'MA Deviation']
        outlier_counts = [outliers_iqr.sum(), outliers_zscore.sum(), outliers_ma.sum()]
        axes[1, 1].bar(outlier_methods, outlier_counts, color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
        axes[1, 1].set_title('Outlier Count by Detection Method', fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(outlier_counts):
            axes[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 09_outlier_detection.png")
        print(f"  IQR法で検出された異常値: {outliers_iqr.sum()} 件")
        print(f"  Z-score法で検出された異常値: {outliers_zscore.sum()} 件")
        
    def calendar_patterns(self, output_dir):
        """曜日・月別パターン詳細分析"""
        print("\n【12. カレンダーパターン分析】")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 曜日×月のヒートマップ
        pivot_table = self.df.pivot_table(values='y', index='dayofweek', columns='month', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Average Call Volume: Day of Week vs Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Day of Week')
        axes[0, 0].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        
        # 週内変動
        dayofweek_stats = self.df.groupby('dayofweek')['y'].agg(['mean', 'std'])
        x = range(7)
        axes[0, 1].bar(x, dayofweek_stats['mean'], yerr=dayofweek_stats['std'], 
                      capsize=5, alpha=0.7, color='steelblue')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].set_title('Call Volume by Day of Week (Mean ± Std)', fontweight='bold')
        axes[0, 1].set_ylabel('Call Volume')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 月内変動（日付別）
        day_stats = self.df.groupby('day')['y'].agg(['mean', 'std'])
        axes[1, 0].bar(range(1, len(day_stats) + 1), day_stats['mean'], 
                      yerr=day_stats['std'], capsize=3, alpha=0.7, color='coral')
        axes[1, 0].set_title('Call Volume by Day of Month (Mean ± Std)', fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 月初・月末効果
        special_days = self.df.groupby(['is_month_start', 'is_month_end'])['y'].mean()
        labels = ['Regular', 'Month Start', 'Month End']
        values = [
            self.df[(self.df['is_month_start'] == 0) & (self.df['is_month_end'] == 0)]['y'].mean(),
            self.df[self.df['is_month_start'] == 1]['y'].mean(),
            self.df[self.df['is_month_end'] == 1]['y'].mean()
        ]
        axes[1, 1].bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        axes[1, 1].set_title('Month Start/End Effect', fontweight='bold')
        axes[1, 1].set_ylabel('Average Call Volume')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_calendar_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 10_calendar_patterns.png")
        
    def variability_analysis(self, output_dir):
        """変動性分析"""
        print("\n【13. 変動性分析】")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ローリング標準偏差
        rolling_std = self.df['y'].rolling(window=30).std()
        axes[0, 0].plot(self.df.index, rolling_std, linewidth=1.5)
        axes[0, 0].set_title('Rolling Standard Deviation (30-day window)', fontweight='bold')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ローリング変動係数
        rolling_mean = self.df['y'].rolling(window=30).mean()
        rolling_cv = rolling_std / rolling_mean
        axes[0, 1].plot(self.df.index, rolling_cv, linewidth=1.5, color='orange')
        axes[0, 1].set_title('Rolling Coefficient of Variation (30-day window)', fontweight='bold')
        axes[0, 1].set_ylabel('CV')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 月別変動係数
        monthly_cv = self.df.groupby('month')['y'].agg(lambda x: x.std() / x.mean())
        axes[1, 0].bar(range(1, 13), monthly_cv.values, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_title('Coefficient of Variation by Month', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('CV')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 曜日別変動係数
        dayofweek_cv = self.df.groupby('dayofweek')['y'].agg(lambda x: x.std() / x.mean())
        axes[1, 1].bar(range(7), dayofweek_cv.values, color='mediumpurple', alpha=0.7)
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[1, 1].set_title('Coefficient of Variation by Day of Week', fontweight='bold')
        axes[1, 1].set_ylabel('CV')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_variability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 11_variability_analysis.png")
        
    def heatmap_analysis(self, output_dir):
        """ヒートマップ分析"""
        print("\n【14. ヒートマップ分析】")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 年×月ヒートマップ
        pivot_year_month = self.df.pivot_table(values='y', index='year', columns='month', aggfunc='mean')
        sns.heatmap(pivot_year_month, annot=True, fmt='.0f', cmap='coolwarm', ax=axes[0, 0], cbar_kws={'label': 'Call Volume'})
        axes[0, 0].set_title('Heatmap: Year vs Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Year')
        
        # 年×曜日ヒートマップ
        pivot_year_dow = self.df.pivot_table(values='y', index='year', columns='dayofweek', aggfunc='mean')
        sns.heatmap(pivot_year_dow, annot=True, fmt='.0f', cmap='viridis', ax=axes[0, 1], cbar_kws={'label': 'Call Volume'})
        axes[0, 1].set_title('Heatmap: Year vs Day of Week', fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Year')
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # 週×曜日ヒートマップ（最近1年）
        recent_year = self.df[self.df['year'] == self.df['year'].max()]
        pivot_week_dow = recent_year.pivot_table(values='y', index='weekofyear', columns='dayofweek', aggfunc='mean')
        sns.heatmap(pivot_week_dow, cmap='RdYlGn_r', ax=axes[1, 0], cbar_kws={'label': 'Call Volume'})
        axes[1, 0].set_title(f'Heatmap: Week vs Day of Week ({recent_year["year"].iloc[0]})', fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Week of Year')
        axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # 四半期×月ヒートマップ
        pivot_quarter_month = self.df.pivot_table(values='y', index='quarter', columns='month', aggfunc='mean')
        sns.heatmap(pivot_quarter_month, annot=True, fmt='.0f', cmap='plasma', ax=axes[1, 1], cbar_kws={'label': 'Call Volume'})
        axes[1, 1].set_title('Heatmap: Quarter vs Month', fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Quarter')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_heatmap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 12_heatmap_analysis.png")
        
    def lag_plots(self, output_dir):
        """ラグプロット"""
        print("\n【15. ラグプロット分析】")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        lags = [1, 2, 3, 7, 14, 30]
        
        for i, lag in enumerate(lags):
            pd.plotting.lag_plot(self.df['y'], lag=lag, ax=axes[i])
            axes[i].set_title(f'Lag Plot (lag={lag} days)', fontweight='bold')
            axes[i].set_xlabel(f't')
            axes[i].set_ylabel(f't-{lag}')
            axes[i].grid(True, alpha=0.3)
            
            # 相関係数を計算
            corr = self.df['y'].autocorr(lag=lag)
            axes[i].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/13_lag_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 13_lag_plots.png")
        
    def difference_analysis(self, output_dir):
        """差分分析"""
        print("\n【16. 差分分析】")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 原系列
        axes[0, 0].plot(self.df.index, self.df['y'], linewidth=0.8)
        axes[0, 0].set_title('Original Series', fontweight='bold')
        axes[0, 0].set_ylabel('Call Volume')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 原系列のヒストグラム
        axes[0, 1].hist(self.df['y'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Original Series Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Call Volume')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 1次差分
        diff_1 = self.df['y'].diff()
        axes[1, 0].plot(self.df.index, diff_1, linewidth=0.8)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('First Difference', fontweight='bold')
        axes[1, 0].set_ylabel('Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(diff_1.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('First Difference Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Difference')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 季節差分（7日）
        diff_seasonal = self.df['y'].diff(7)
        axes[2, 0].plot(self.df.index, diff_seasonal, linewidth=0.8)
        axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2, 0].set_title('Seasonal Difference (7 days)', fontweight='bold')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Difference')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].hist(diff_seasonal.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[2, 1].set_title('Seasonal Difference Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('Difference')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/14_difference_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 14_difference_analysis.png")


# =============================================================================
# メイン実行例
# =============================================================================

if __name__ == "__main__":
    """
    使用例:
    
    # CSVファイルからデータを読み込み
    df = pd.read_csv('your_data.csv')
    
    # EDA実行
    eda = CallCenterEDA(df)
    eda.run_all_analyses(output_dir='./eda_results')
    """
    
    # サンプルデータでのデモ
    print("\n" + "="*80)
    print("このスクリプトを使用するには:")
    print("="*80)
    print("""
    import pandas as pd
    from call_center_eda_comprehensive import CallCenterEDA
    
    # データ読み込み
    df = pd.read_csv('your_data.csv')  # ds, y カラムを含むCSV
    
    # EDA実行
    eda = CallCenterEDA(df)
    eda.run_all_analyses(output_dir='./eda_results')
    """)
    print("="*80)
