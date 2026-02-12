"""
複数モデル予測結果の比較・分析ツール
Prophet形式データ（ds, y）に対応した包括的な比較分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MultiModelComparison:
    """複数モデルの予測結果を比較分析するクラス"""
    
    def __init__(self, model_files, model_names=None, actual_file=None):
        """
        Parameters:
        -----------
        model_files : list of str
            各モデルの予測結果CSVファイルパスのリスト
        model_names : list of str or None
            各モデルの名前リスト。Noneの場合は自動生成
        actual_file : str or None
            実績データのCSVファイルパス（あれば）
        """
        self.model_files = model_files
        self.model_names = model_names if model_names else [f'Model_{i+1}' for i in range(len(model_files))]
        self.actual_file = actual_file
        
        # データ読み込み
        self.models_df = {}
        self.combined_df = None
        self.actual_df = None
        
        self._load_data()
        self._create_combined_df()
        
    def _load_data(self):
        """全モデルのデータを読み込み"""
        print("\n" + "="*80)
        print("データ読み込み")
        print("="*80)
        
        for i, (file, name) in enumerate(zip(self.model_files, self.model_names)):
            try:
                df = pd.read_csv(file)
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.sort_values('ds').reset_index(drop=True)
                self.models_df[name] = df
                print(f"  ✓ {name}: {len(df)} レコード ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
            except Exception as e:
                print(f"  ✗ {name}: エラー - {e}")
        
        # 実績データの読み込み
        if self.actual_file:
            try:
                self.actual_df = pd.read_csv(self.actual_file)
                self.actual_df['ds'] = pd.to_datetime(self.actual_df['ds'])
                self.actual_df = self.actual_df.sort_values('ds').reset_index(drop=True)
                print(f"  ✓ 実績データ: {len(self.actual_df)} レコード")
            except Exception as e:
                print(f"  ✗ 実績データ: エラー - {e}")
        
        print("="*80 + "\n")
    
    def _create_combined_df(self):
        """全モデルの予測を1つのDataFrameに結合"""
        # 最初のモデルをベースに
        first_model = list(self.models_df.keys())[0]
        self.combined_df = self.models_df[first_model][['ds']].copy()
        
        # 各モデルの予測値を追加
        for name, df in self.models_df.items():
            self.combined_df = self.combined_df.merge(
                df[['ds', 'y']].rename(columns={'y': name}),
                on='ds',
                how='outer'
            )
        
        # 実績データを追加
        if self.actual_df is not None:
            self.combined_df = self.combined_df.merge(
                self.actual_df[['ds', 'y']].rename(columns={'y': 'Actual'}),
                on='ds',
                how='left'
            )
        
        self.combined_df = self.combined_df.sort_values('ds').reset_index(drop=True)
        
        # 追加特徴量
        self.combined_df['dayofweek'] = self.combined_df['ds'].dt.dayofweek
        self.combined_df['month'] = self.combined_df['ds'].dt.month
        self.combined_df['day'] = self.combined_df['ds'].dt.day
        self.combined_df['weekofyear'] = self.combined_df['ds'].dt.isocalendar().week
        self.combined_df['is_weekend'] = (self.combined_df['dayofweek'] >= 5).astype(int)
    
    def summary_statistics(self):
        """各モデルの基本統計量"""
        print("\n【1. 基本統計量】")
        print("="*80)
        
        stats_df = pd.DataFrame()
        
        for name in self.model_names:
            if name in self.combined_df.columns:
                data = self.combined_df[name].dropna()
                stats_df[name] = {
                    'Count': len(data),
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    '25%': data.quantile(0.25),
                    'Median': data.median(),
                    '75%': data.quantile(0.75),
                    'Max': data.max(),
                    'CV': data.std() / data.mean(),
                    'Skewness': data.skew(),
                    'Kurtosis': data.kurtosis()
                }
        
        print(stats_df.T.to_string())
        print("="*80 + "\n")
        
        return stats_df.T
    
    def correlation_analysis(self, output_dir='./comparison_output'):
        """相関分析"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【2. 相関分析】")
        print("="*80)
        
        # 相関行列を計算
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            model_cols.append('Actual')
        
        corr_matrix = self.combined_df[model_cols].corr()
        
        print("相関行列:")
        print(corr_matrix.to_string())
        print("\n")
        
        # 相関行列のヒートマップ
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ヒートマップ1: 数値表示
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                   cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Correlation Matrix (All Models)', fontsize=14, fontweight='bold')
        
        # ヒートマップ2: クラスターマップ風
        sns.heatmap(corr_matrix, cmap='RdYlGn', center=0, vmin=-1, vmax=1, 
                   square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Correlation Matrix (Color Coded)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 01_correlation_matrix.png")
        
        # 相関係数のペアワイズ比較
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 上三角の相関係数を取得
        correlations = []
        pairs = []
        for i in range(len(model_cols)):
            for j in range(i+1, len(model_cols)):
                correlations.append(corr_matrix.iloc[i, j])
                pairs.append(f"{model_cols[i]}\nvs\n{model_cols[j]}")
        
        colors = ['green' if c > 0.95 else 'orange' if c > 0.9 else 'red' for c in correlations]
        bars = ax.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_xlabel('Correlation Coefficient', fontweight='bold')
        ax.set_title('Pairwise Model Correlations', fontsize=14, fontweight='bold')
        ax.axvline(x=0.9, color='orange', linestyle='--', alpha=0.5, label='0.9 threshold')
        ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, label='0.95 threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_pairwise_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 02_pairwise_correlations.png")
        
        return corr_matrix
    
    def plot_timeseries_comparison(self, output_dir='./comparison_output'):
        """時系列比較プロット"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【3. 時系列比較プロット】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # プロット1: 全モデル重ね描き
        for name in model_cols:
            axes[0].plot(self.combined_df['ds'], self.combined_df[name], 
                        label=name, linewidth=1.5, alpha=0.8)
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            axes[0].plot(self.combined_df['ds'], self.combined_df['Actual'], 
                        label='Actual', linewidth=2, color='black', linestyle='--', alpha=0.9)
        
        axes[0].set_title('All Models Forecast Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Forecasted Value')
        axes[0].legend(loc='best', ncol=3)
        axes[0].grid(True, alpha=0.3)
        
        # プロット2: 平均と標準偏差
        model_mean = self.combined_df[model_cols].mean(axis=1)
        model_std = self.combined_df[model_cols].std(axis=1)
        
        axes[1].plot(self.combined_df['ds'], model_mean, 
                    label='Mean of Models', linewidth=2, color='blue')
        axes[1].fill_between(self.combined_df['ds'], 
                            model_mean - model_std, 
                            model_mean + model_std,
                            alpha=0.3, color='blue', label='±1 Std Dev')
        axes[1].fill_between(self.combined_df['ds'], 
                            model_mean - 2*model_std, 
                            model_mean + 2*model_std,
                            alpha=0.15, color='blue', label='±2 Std Dev')
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            axes[1].plot(self.combined_df['ds'], self.combined_df['Actual'], 
                        label='Actual', linewidth=2, color='red', linestyle='--')
        
        axes[1].set_title('Model Ensemble (Mean ± Std)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # プロット3: モデル間のばらつき（変動係数）
        model_cv = model_std / model_mean
        axes[2].plot(self.combined_df['ds'], model_cv, 
                    linewidth=1.5, color='purple')
        axes[2].fill_between(self.combined_df['ds'], 0, model_cv, alpha=0.3, color='purple')
        axes[2].axhline(y=model_cv.mean(), color='red', linestyle='--', 
                       label=f'Mean CV: {model_cv.mean():.4f}')
        axes[2].set_title('Model Disagreement (Coefficient of Variation)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('CV (Std/Mean)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_timeseries_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 03_timeseries_comparison.png")
    
    def plot_distribution_comparison(self, output_dir='./comparison_output'):
        """分布比較"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【4. 分布比較】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ヒストグラム
        for name in model_cols:
            axes[0, 0].hist(self.combined_df[name].dropna(), bins=30, 
                          alpha=0.5, label=name, edgecolor='black')
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            axes[0, 0].hist(self.combined_df['Actual'].dropna(), bins=30, 
                          alpha=0.7, label='Actual', edgecolor='black', color='red')
        
        axes[0, 0].set_title('Distribution Comparison (Histogram)', fontweight='bold')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # KDE
        for name in model_cols:
            self.combined_df[name].dropna().plot(kind='kde', ax=axes[0, 1], 
                                                label=name, linewidth=2)
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            self.combined_df['Actual'].dropna().plot(kind='kde', ax=axes[0, 1], 
                                                    label='Actual', linewidth=2.5, 
                                                    color='red', linestyle='--')
        
        axes[0, 1].set_title('Density Comparison (KDE)', fontweight='bold')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        box_data = [self.combined_df[name].dropna() for name in model_cols]
        box_labels = model_cols
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            box_data.append(self.combined_df['Actual'].dropna())
            box_labels.append('Actual')
        
        bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # ボックスに色を付ける
        colors = plt.cm.Set3(range(len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 0].set_title('Box Plot Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Violin plot
        violin_data = pd.DataFrame({name: self.combined_df[name].dropna().values 
                                   for name in model_cols})
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            actual_data = self.combined_df['Actual'].dropna()
            if len(actual_data) > 0:
                violin_data['Actual'] = actual_data.values[:len(violin_data)]
        
        violin_data.plot(kind='violin', ax=axes[1, 1])
        axes[1, 1].set_title('Violin Plot Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 04_distribution_comparison.png")
    
    def scatter_matrix(self, output_dir='./comparison_output'):
        """散布図行列"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【5. 散布図行列】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            model_cols.append('Actual')
        
        # ペアプロット
        plot_df = self.combined_df[model_cols].dropna()
        
        if len(plot_df) > 0:
            g = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                           diag_kws={'linewidth': 2})
            g.fig.suptitle('Scatter Matrix - All Models', fontsize=16, fontweight='bold', y=1.01)
            plt.savefig(f'{output_dir}/05_scatter_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  保存: 05_scatter_matrix.png")
        else:
            print("  スキップ: データが不足しています")
    
    def difference_analysis(self, output_dir='./comparison_output'):
        """モデル間の差分分析"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【6. モデル間差分分析】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        if len(model_cols) < 2:
            print("  スキップ: モデルが2つ以上必要です")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 最初のモデルとの差分
        base_model = model_cols[0]
        
        for name in model_cols[1:]:
            diff = self.combined_df[name] - self.combined_df[base_model]
            axes[0, 0].plot(self.combined_df['ds'], diff, label=f'{name} - {base_model}', 
                          linewidth=1.5, alpha=0.7)
        
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'Difference from {base_model}', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Difference')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # モデル間の最大差分
        model_max = self.combined_df[model_cols].max(axis=1)
        model_min = self.combined_df[model_cols].min(axis=1)
        model_range = model_max - model_min
        
        axes[0, 1].plot(self.combined_df['ds'], model_range, linewidth=1.5, color='purple')
        axes[0, 1].fill_between(self.combined_df['ds'], 0, model_range, alpha=0.3, color='purple')
        axes[0, 1].set_title('Range (Max - Min) Across Models', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Range')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 差分の分布
        for name in model_cols[1:]:
            diff = self.combined_df[name] - self.combined_df[base_model]
            axes[1, 0].hist(diff.dropna(), bins=30, alpha=0.5, 
                          label=f'{name} - {base_model}', edgecolor='black')
        
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('Distribution of Differences', fontweight='bold')
        axes[1, 0].set_xlabel('Difference')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # ペアワイズ差分の絶対値平均
        diff_matrix = np.zeros((len(model_cols), len(model_cols)))
        
        for i, model1 in enumerate(model_cols):
            for j, model2 in enumerate(model_cols):
                if i != j:
                    diff = np.abs(self.combined_df[model1] - self.combined_df[model2])
                    diff_matrix[i, j] = diff.mean()
        
        sns.heatmap(diff_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=model_cols, yticklabels=model_cols,
                   ax=axes[1, 1], cbar_kws={'label': 'Mean Absolute Difference'})
        axes[1, 1].set_title('Pairwise Mean Absolute Difference', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_difference_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 06_difference_analysis.png")
    
    def calendar_pattern_comparison(self, output_dir='./comparison_output'):
        """曜日・月別パターンの比較"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【7. カレンダーパターン比較】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 曜日別平均
        for name in model_cols:
            dow_mean = self.combined_df.groupby('dayofweek')[name].mean()
            axes[0, 0].plot(range(7), dow_mean.values, marker='o', linewidth=2, 
                          markersize=8, label=name, alpha=0.7)
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            dow_mean_actual = self.combined_df.groupby('dayofweek')['Actual'].mean()
            axes[0, 0].plot(range(7), dow_mean_actual.values, marker='s', linewidth=2.5, 
                          markersize=10, label='Actual', color='red', linestyle='--')
        
        axes[0, 0].set_xticks(range(7))
        axes[0, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 0].set_title('Average by Day of Week', fontweight='bold')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 月別平均
        for name in model_cols:
            month_mean = self.combined_df.groupby('month')[name].mean()
            axes[0, 1].plot(month_mean.index, month_mean.values, marker='o', 
                          linewidth=2, markersize=8, label=name, alpha=0.7)
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            month_mean_actual = self.combined_df.groupby('month')['Actual'].mean()
            axes[0, 1].plot(month_mean_actual.index, month_mean_actual.values, 
                          marker='s', linewidth=2.5, markersize=10, label='Actual', 
                          color='red', linestyle='--')
        
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_title('Average by Month', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 曜日別の変動係数
        for name in model_cols:
            dow_cv = self.combined_df.groupby('dayofweek')[name].agg(lambda x: x.std() / x.mean())
            axes[1, 0].bar(np.arange(7) + (model_cols.index(name) * 0.15), 
                          dow_cv.values, width=0.15, label=name, alpha=0.7)
        
        axes[1, 0].set_xticks(np.arange(7) + 0.15 * (len(model_cols) - 1) / 2)
        axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[1, 0].set_title('Coefficient of Variation by Day of Week', fontweight='bold')
        axes[1, 0].set_ylabel('CV')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 週末vs平日
        weekend_comparison = {}
        for name in model_cols:
            weekend_comparison[name] = {
                'Weekday': self.combined_df[self.combined_df['is_weekend'] == 0][name].mean(),
                'Weekend': self.combined_df[self.combined_df['is_weekend'] == 1][name].mean()
            }
        
        weekend_df = pd.DataFrame(weekend_comparison).T
        weekend_df.plot(kind='bar', ax=axes[1, 1], alpha=0.7, rot=45)
        axes[1, 1].set_title('Weekday vs Weekend Average', fontweight='bold')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_calendar_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 07_calendar_patterns.png")
    
    def model_agreement_analysis(self, output_dir='./comparison_output'):
        """モデル一致度分析"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【8. モデル一致度分析】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        if len(model_cols) < 2:
            print("  スキップ: モデルが2つ以上必要です")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 標準偏差の時系列
        model_std = self.combined_df[model_cols].std(axis=1)
        axes[0, 0].plot(self.combined_df['ds'], model_std, linewidth=1.5, color='blue')
        axes[0, 0].fill_between(self.combined_df['ds'], 0, model_std, alpha=0.3, color='blue')
        axes[0, 0].set_title('Model Standard Deviation Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Std Dev')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 変動係数の時系列
        model_mean = self.combined_df[model_cols].mean(axis=1)
        model_cv = model_std / model_mean
        axes[0, 1].plot(self.combined_df['ds'], model_cv, linewidth=1.5, color='green')
        axes[0, 1].fill_between(self.combined_df['ds'], 0, model_cv, alpha=0.3, color='green')
        axes[0, 1].axhline(y=model_cv.mean(), color='red', linestyle='--', 
                          label=f'Mean: {model_cv.mean():.4f}')
        axes[0, 1].set_title('Model Coefficient of Variation Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('CV')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 一致度スコア（全モデルが近い値を出しているか）
        # 範囲が小さいほど一致度が高い
        model_range = self.combined_df[model_cols].max(axis=1) - self.combined_df[model_cols].min(axis=1)
        agreement_score = 1 / (1 + model_range)  # 0~1のスコア
        
        axes[1, 0].plot(self.combined_df['ds'], agreement_score, linewidth=1.5, color='purple')
        axes[1, 0].fill_between(self.combined_df['ds'], 0, agreement_score, alpha=0.3, color='purple')
        axes[1, 0].axhline(y=agreement_score.mean(), color='red', linestyle='--', 
                          label=f'Mean: {agreement_score.mean():.4f}')
        axes[1, 0].set_title('Model Agreement Score (higher = more agreement)', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Agreement Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 一致度の分布
        axes[1, 1].hist(agreement_score, bins=30, edgecolor='black', alpha=0.7, color='teal')
        axes[1, 1].axvline(x=agreement_score.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {agreement_score.mean():.4f}')
        axes[1, 1].set_title('Distribution of Agreement Score', fontweight='bold')
        axes[1, 1].set_xlabel('Agreement Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_model_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 08_model_agreement.png")
    
    def rank_analysis(self, output_dir='./comparison_output'):
        """ランク分析（各時点でどのモデルが最大/最小か）"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【9. ランク分析】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        if len(model_cols) < 2:
            print("  スキップ: モデルが2つ以上必要です")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 各時点でのランク
        rank_df = self.combined_df[model_cols].rank(axis=1, ascending=False)
        
        # ランク1（最大値）の頻度
        rank1_counts = (rank_df == 1).sum()
        axes[0, 0].bar(range(len(model_cols)), rank1_counts.values, 
                      color=plt.cm.Set3(range(len(model_cols))), alpha=0.7)
        axes[0, 0].set_xticks(range(len(model_cols)))
        axes[0, 0].set_xticklabels(model_cols, rotation=45)
        axes[0, 0].set_title('Frequency of Being Highest Forecast', fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(rank1_counts.values):
            axes[0, 0].text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
        
        # 平均ランク
        mean_rank = rank_df.mean()
        axes[0, 1].bar(range(len(model_cols)), mean_rank.values, 
                      color=plt.cm.Set2(range(len(model_cols))), alpha=0.7)
        axes[0, 1].set_xticks(range(len(model_cols)))
        axes[0, 1].set_xticklabels(model_cols, rotation=45)
        axes[0, 1].set_title('Average Rank (lower = higher forecasts)', fontweight='bold')
        axes[0, 1].set_ylabel('Average Rank')
        axes[0, 1].invert_yaxis()  # 低いランクが上に
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(mean_rank.values):
            axes[0, 1].text(i, v - 0.05, f'{v:.2f}', ha='center', fontweight='bold')
        
        # ランクの時系列
        for name in model_cols:
            axes[1, 0].plot(self.combined_df['ds'], rank_df[name], 
                          label=name, linewidth=1.5, alpha=0.7)
        
        axes[1, 0].set_title('Rank Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Rank')
        axes[1, 0].legend()
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3)
        
        # ランクの安定性（標準偏差）
        rank_stability = rank_df.std()
        axes[1, 1].bar(range(len(model_cols)), rank_stability.values, 
                      color=plt.cm.Pastel1(range(len(model_cols))), alpha=0.7)
        axes[1, 1].set_xticks(range(len(model_cols)))
        axes[1, 1].set_xticklabels(model_cols, rotation=45)
        axes[1, 1].set_title('Rank Stability (Std Dev, lower = more stable)', fontweight='bold')
        axes[1, 1].set_ylabel('Std Dev of Rank')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(rank_stability.values):
            axes[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_rank_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 09_rank_analysis.png")
    
    def ensemble_forecast(self, output_dir='./comparison_output'):
        """アンサンブル予測（加重平均など）"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【10. アンサンブル予測】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        # 単純平均
        self.combined_df['Ensemble_Mean'] = self.combined_df[model_cols].mean(axis=1)
        
        # 中央値
        self.combined_df['Ensemble_Median'] = self.combined_df[model_cols].median(axis=1)
        
        # トリム平均（上下10%を除外）
        def trimmed_mean(row):
            values = row.dropna().values
            if len(values) < 3:
                return np.mean(values)
            trim_count = max(1, int(len(values) * 0.1))
            sorted_values = np.sort(values)
            return np.mean(sorted_values[trim_count:-trim_count])
        
        self.combined_df['Ensemble_Trimmed'] = self.combined_df[model_cols].apply(trimmed_mean, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # アンサンブル手法の比較
        ensemble_methods = ['Ensemble_Mean', 'Ensemble_Median', 'Ensemble_Trimmed']
        
        for method in ensemble_methods:
            axes[0, 0].plot(self.combined_df['ds'], self.combined_df[method], 
                          label=method.replace('Ensemble_', ''), linewidth=2, alpha=0.7)
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            axes[0, 0].plot(self.combined_df['ds'], self.combined_df['Actual'], 
                          label='Actual', linewidth=2.5, color='red', linestyle='--')
        
        axes[0, 0].set_title('Ensemble Methods Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 個別モデルとアンサンブル平均
        for name in model_cols:
            axes[0, 1].plot(self.combined_df['ds'], self.combined_df[name], 
                          label=name, linewidth=1, alpha=0.4)
        
        axes[0, 1].plot(self.combined_df['ds'], self.combined_df['Ensemble_Mean'], 
                       label='Ensemble Mean', linewidth=2.5, color='blue')
        
        if self.actual_df is not None and 'Actual' in self.combined_df.columns:
            axes[0, 1].plot(self.combined_df['ds'], self.combined_df['Actual'], 
                          label='Actual', linewidth=2.5, color='red', linestyle='--')
        
        axes[0, 1].set_title('Individual Models vs Ensemble', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # アンサンブル手法間の差分
        diff_mean_median = self.combined_df['Ensemble_Mean'] - self.combined_df['Ensemble_Median']
        diff_mean_trimmed = self.combined_df['Ensemble_Mean'] - self.combined_df['Ensemble_Trimmed']
        
        axes[1, 0].plot(self.combined_df['ds'], diff_mean_median, 
                       label='Mean - Median', linewidth=1.5)
        axes[1, 0].plot(self.combined_df['ds'], diff_mean_trimmed, 
                       label='Mean - Trimmed', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Difference Between Ensemble Methods', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Difference')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # アンサンブルの統計
        ensemble_stats = pd.DataFrame({
            'Mean': [self.combined_df['Ensemble_Mean'].mean(),
                    self.combined_df['Ensemble_Mean'].std(),
                    self.combined_df['Ensemble_Mean'].min(),
                    self.combined_df['Ensemble_Mean'].max()],
            'Median': [self.combined_df['Ensemble_Median'].mean(),
                      self.combined_df['Ensemble_Median'].std(),
                      self.combined_df['Ensemble_Median'].min(),
                      self.combined_df['Ensemble_Median'].max()],
            'Trimmed': [self.combined_df['Ensemble_Trimmed'].mean(),
                       self.combined_df['Ensemble_Trimmed'].std(),
                       self.combined_df['Ensemble_Trimmed'].min(),
                       self.combined_df['Ensemble_Trimmed'].max()]
        }, index=['Mean', 'Std', 'Min', 'Max'])
        
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=ensemble_stats.round(2).values,
                                rowLabels=ensemble_stats.index,
                                colLabels=ensemble_stats.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Ensemble Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_ensemble_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 10_ensemble_forecast.png")
        
        # アンサンブル結果を保存
        ensemble_output = self.combined_df[['ds'] + ensemble_methods].copy()
        ensemble_output.to_csv(f'{output_dir}/ensemble_forecasts.csv', index=False)
        print(f"  アンサンブル予測を保存: ensemble_forecasts.csv")
    
    def clustering_analysis(self, output_dir='./comparison_output'):
        """モデルのクラスタリング分析"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【11. クラスタリング分析】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        if len(model_cols) < 3:
            print("  スキップ: モデルが3つ以上必要です")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 階層的クラスタリング（モデル間）
        corr_matrix = self.combined_df[model_cols].corr()
        distance_matrix = 1 - corr_matrix
        
        linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
        dendro = hierarchy.dendrogram(linkage_matrix, labels=model_cols, ax=axes[0])
        axes[0].set_title('Hierarchical Clustering of Models', fontweight='bold')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Distance')
        
        # クラスターマップ
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.spatial.distance as ssd
        
        # 距離行列を1次元配列に変換
        distArray = ssd.squareform(distance_matrix)
        
        # クラスタリング実行
        linkage_matrix2 = linkage(distArray, method='ward')
        
        # デンドログラムから順序を取得
        dendro_order = hierarchy.dendrogram(linkage_matrix2, no_plot=True)['leaves']
        
        # 順序に従って並び替え
        ordered_corr = corr_matrix.iloc[dendro_order, dendro_order]
        
        sns.heatmap(ordered_corr, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[1],
                   cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Clustered Correlation Matrix', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 11_clustering_analysis.png")
    
    def performance_metrics_comparison(self, output_dir='./comparison_output'):
        """実績データがある場合の性能比較"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.actual_df is None or 'Actual' not in self.combined_df.columns:
            print("\n【12. 性能比較】")
            print("  スキップ: 実績データがありません")
            return
        
        print("\n【12. 性能比較（実績データあり）】")
        
        model_cols = [col for col in self.model_names if col in self.combined_df.columns]
        
        # メトリクス計算
        metrics_data = []
        
        for name in model_cols:
            # 欠損値を除外
            mask = self.combined_df[[name, 'Actual']].notna().all(axis=1)
            y_true = self.combined_df.loc[mask, 'Actual'].values
            y_pred = self.combined_df.loc[mask, name].values
            
            if len(y_true) > 0:
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # バイアス
                bias = np.mean(y_pred - y_true)
                
                metrics_data.append({
                    'Model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'Bias': bias,
                    'N': len(y_true)
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        print("\n性能メトリクス:")
        print(metrics_df.to_string(index=False))
        
        # メトリクスを保存
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)
        print(f"\n  保存: performance_metrics.csv")
        
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE
        axes[0, 0].bar(range(len(metrics_df)), metrics_df['MAE'], 
                      color=plt.cm.Set3(range(len(metrics_df))), alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics_df)))
        axes[0, 0].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(metrics_df['MAE']):
            axes[0, 0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')
        
        # RMSE
        axes[0, 1].bar(range(len(metrics_df)), metrics_df['RMSE'], 
                      color=plt.cm.Set2(range(len(metrics_df))), alpha=0.7)
        axes[0, 1].set_xticks(range(len(metrics_df)))
        axes[0, 1].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(metrics_df['RMSE']):
            axes[0, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')
        
        # MAPE
        axes[1, 0].bar(range(len(metrics_df)), metrics_df['MAPE'], 
                      color=plt.cm.Pastel1(range(len(metrics_df))), alpha=0.7)
        axes[1, 0].set_xticks(range(len(metrics_df)))
        axes[1, 0].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)', fontweight='bold')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(metrics_df['MAPE']):
            axes[1, 0].text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Bias
        colors = ['green' if b < 0 else 'red' for b in metrics_df['Bias']]
        axes[1, 1].bar(range(len(metrics_df)), metrics_df['Bias'], 
                      color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_xticks(range(len(metrics_df)))
        axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 1].set_title('Bias (Positive = Over-forecast)', fontweight='bold')
        axes[1, 1].set_ylabel('Bias')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(metrics_df['Bias']):
            axes[1, 1].text(i, v + (0.5 if v > 0 else -1), f'{v:.2f}', 
                          ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: 12_performance_metrics.png")
        
        return metrics_df
    
    def run_full_comparison(self, output_dir='./comparison_output'):
        """全ての比較分析を実行"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("複数モデル比較分析開始")
        print("="*80)
        
        # 1. 基本統計量
        stats_df = self.summary_statistics()
        stats_df.to_csv(f'{output_dir}/summary_statistics.csv')
        
        # 2. 相関分析
        corr_matrix = self.correlation_analysis(output_dir)
        corr_matrix.to_csv(f'{output_dir}/correlation_matrix.csv')
        
        # 3. 時系列比較
        self.plot_timeseries_comparison(output_dir)
        
        # 4. 分布比較
        self.plot_distribution_comparison(output_dir)
        
        # 5. 散布図行列
        self.scatter_matrix(output_dir)
        
        # 6. 差分分析
        self.difference_analysis(output_dir)
        
        # 7. カレンダーパターン
        self.calendar_pattern_comparison(output_dir)
        
        # 8. モデル一致度
        self.model_agreement_analysis(output_dir)
        
        # 9. ランク分析
        self.rank_analysis(output_dir)
        
        # 10. アンサンブル予測
        self.ensemble_forecast(output_dir)
        
        # 11. クラスタリング
        self.clustering_analysis(output_dir)
        
        # 12. 性能比較（実績データがあれば）
        if self.actual_df is not None:
            metrics_df = self.performance_metrics_comparison(output_dir)
        
        # 統合レポート用のDataFrameを保存
        self.combined_df.to_csv(f'{output_dir}/combined_forecasts.csv', index=False)
        print(f"\n統合データを保存: combined_forecasts.csv")
        
        print("\n" + "="*80)
        print("比較分析完了!")
        print(f"全ての結果は '{output_dir}' に保存されています")
        print("="*80)


# =============================================================================
# メイン実行例
# =============================================================================

if __name__ == "__main__":
    """
    使用例:
    
    # 複数モデルの予測結果を比較
    model_files = [
        'model1_forecast.csv',
        'model2_forecast.csv',
        'model3_forecast.csv',
        'model4_forecast.csv',
        'model5_forecast.csv'
    ]
    
    model_names = ['SARIMAX', 'Prophet', 'LSTM', 'XGBoost', 'Random Forest']
    
    # 実績データがあれば
    comparison = MultiModelComparison(
        model_files=model_files,
        model_names=model_names,
        actual_file='actual_data.csv'  # オプション
    )
    
    # 全ての比較分析を実行
    comparison.run_full_comparison(output_dir='./model_comparison_results')
    """
    
    print("\n" + "="*80)
    print("このスクリプトを使用するには:")
    print("="*80)
    print("""
    from multi_model_comparison_analysis import MultiModelComparison
    
    # 複数モデルのCSVファイルパス
    model_files = [
        'model1_forecast.csv',
        'model2_forecast.csv',
        'model3_forecast.csv',
        'model4_forecast.csv',
        'model5_forecast.csv'
    ]
    
    # モデル名（オプション）
    model_names = ['SARIMAX', 'Prophet', 'LSTM', 'XGBoost', 'Random Forest']
    
    # 比較分析実行
    comparison = MultiModelComparison(
        model_files=model_files,
        model_names=model_names,
        actual_file='actual_data.csv'  # 実績データ（あれば）
    )
    
    # 全分析実行
    comparison.run_full_comparison(output_dir='./comparison_results')
    
    # 個別に実行することも可能
    # comparison.correlation_analysis()
    # comparison.plot_timeseries_comparison()
    # comparison.ensemble_forecast()
    """)
    print("="*80)
