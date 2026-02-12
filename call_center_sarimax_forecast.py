"""
コールセンター呼量予測 - SARIMAXモデル
Prophet形式データ（ds, y）に対応した包括的な予測システム
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse, meanabs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CallCenterSARIMAX:
    """コールセンター呼量予測 - SARIMAXモデルクラス"""
    
    def __init__(self, df, test_size=30):
        """
        Parameters:
        -----------
        df : DataFrame
            'ds' (日付) と 'y' (呼量) のカラムを持つDataFrame
        test_size : int
            テストデータのサイズ（日数）
        """
        self.df = df.copy()
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df = self.df.sort_values('ds').reset_index(drop=True)
        self.df.set_index('ds', inplace=True)
        
        self.test_size = test_size
        self.train = None
        self.test = None
        self.model = None
        self.model_fit = None
        self.best_params = None
        
        # データ分割
        self._split_data()
        
        # 外生変数の作成
        self._create_exog_features()
        
    def _split_data(self):
        """訓練データとテストデータに分割"""
        self.train = self.df.iloc[:-self.test_size]
        self.test = self.df.iloc[-self.test_size:]
        
        print(f"\n{'='*80}")
        print(f"データ分割:")
        print(f"  訓練データ: {self.train.index.min().date()} ~ {self.train.index.max().date()} ({len(self.train)}日)")
        print(f"  テストデータ: {self.test.index.min().date()} ~ {self.test.index.max().date()} ({len(self.test)}日)")
        print(f"{'='*80}\n")
        
    def _create_exog_features(self):
        """外生変数（曜日、月、祝日など）を作成"""
        for data in [self.train, self.test]:
            data['dayofweek'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['day'] = data.index.day
            data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
            data['is_month_start'] = data.index.is_month_start.astype(int)
            data['is_month_end'] = data.index.is_month_end.astype(int)
            
            # 曜日ダミー変数
            for i in range(7):
                data[f'dow_{i}'] = (data['dayofweek'] == i).astype(int)
            
            # 月ダミー変数
            for i in range(1, 13):
                data[f'month_{i}'] = (data['month'] == i).astype(int)
    
    def check_stationarity(self):
        """定常性のチェック（ADF検定）"""
        print("\n【定常性チェック - ADF検定】")
        print("-" * 80)
        
        result = adfuller(self.train['y'].dropna())
        print(f"ADF統計量: {result[0]:.6f}")
        print(f"p値: {result[1]:.6f}")
        print(f"臨界値:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")
        
        if result[1] <= 0.05:
            print("\n結論: データは定常です (p <= 0.05)")
            return True
        else:
            print("\n結論: データは非定常です (p > 0.05)")
            print("      差分を取る必要があるかもしれません (dパラメータ)")
            return False
    
    def plot_acf_pacf(self, output_dir='./sarimax_output'):
        """ACF/PACFプロットでパラメータ推定の参考にする"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【ACF/PACFプロット】")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # ACF
        plot_acf(self.train['y'].dropna(), lags=60, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Lag (days)')
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(self.train['y'].dropna(), lags=60, ax=axes[1], alpha=0.05)
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Lag (days)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/acf_pacf_for_params.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: acf_pacf_for_params.png")
        
    def auto_arima_search(self, output_dir='./sarimax_output', seasonal=True, 
                          p_range=(0, 3), d_range=(0, 2), q_range=(0, 3),
                          P_range=(0, 2), D_range=(0, 2), Q_range=(0, 2),
                          m=7, use_exog=True, max_iterations=50):
        """
        グリッドサーチで最適なSARIMAXパラメータを探索
        
        Parameters:
        -----------
        seasonal : bool
            季節性を考慮するか
        p_range, d_range, q_range : tuple
            非季節ARIMAのパラメータ範囲
        P_range, D_range, Q_range : tuple
            季節ARIMAのパラメータ範囲
        m : int
            季節周期（7=週次、30=月次）
        use_exog : bool
            外生変数を使用するか
        max_iterations : int
            探索する組み合わせの最大数
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【SARIMAXパラメータ自動探索】")
        print("-" * 80)
        print(f"探索範囲:")
        print(f"  p: {p_range}, d: {d_range}, q: {q_range}")
        if seasonal:
            print(f"  P: {P_range}, D: {D_range}, Q: {Q_range}, m: {m}")
        print(f"  外生変数使用: {use_exog}")
        print(f"  最大反復回数: {max_iterations}")
        print("-" * 80)
        
        # パラメータの組み合わせを生成
        pdq = list(itertools.product(range(p_range[0], p_range[1]+1),
                                     range(d_range[0], d_range[1]+1),
                                     range(q_range[0], q_range[1]+1)))
        
        if seasonal:
            seasonal_pdq = list(itertools.product(range(P_range[0], P_range[1]+1),
                                                   range(D_range[0], D_range[1]+1),
                                                   range(Q_range[0], Q_range[1]+1),
                                                   [m]))
        else:
            seasonal_pdq = [(0, 0, 0, 0)]
        
        # 外生変数の準備
        if use_exog:
            exog_cols = ['is_weekend', 'is_month_start', 'is_month_end'] + \
                       [f'dow_{i}' for i in range(7)] + \
                       [f'month_{i}' for i in range(1, 13)]
            exog_train = self.train[exog_cols]
        else:
            exog_train = None
        
        results = []
        iteration = 0
        
        print("\n探索開始...")
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                if iteration >= max_iterations:
                    break
                    
                try:
                    iteration += 1
                    model = SARIMAX(self.train['y'],
                                   order=param,
                                   seasonal_order=param_seasonal,
                                   exog=exog_train,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    
                    model_fit = model.fit(disp=False, maxiter=200)
                    
                    results.append({
                        'order': param,
                        'seasonal_order': param_seasonal,
                        'aic': model_fit.aic,
                        'bic': model_fit.bic,
                        'model_fit': model_fit
                    })
                    
                    if iteration % 10 == 0:
                        print(f"  {iteration}個のモデルを評価しました...")
                    
                except Exception as e:
                    continue
                    
            if iteration >= max_iterations:
                break
        
        if not results:
            raise ValueError("有効なモデルが見つかりませんでした")
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame([{
            'order': r['order'],
            'seasonal_order': r['seasonal_order'],
            'AIC': r['aic'],
            'BIC': r['bic']
        } for r in results])
        
        # AICで最良モデルを選択
        best_result = min(results, key=lambda x: x['aic'])
        self.best_params = {
            'order': best_result['order'],
            'seasonal_order': best_result['seasonal_order'],
            'use_exog': use_exog
        }
        self.model_fit = best_result['model_fit']
        
        print(f"\n探索完了: {len(results)}個のモデルを評価")
        print(f"\n最良モデル:")
        print(f"  SARIMA{self.best_params['order']}x{self.best_params['seasonal_order']}")
        print(f"  AIC: {best_result['aic']:.2f}")
        print(f"  BIC: {best_result['bic']:.2f}")
        
        # 上位10モデルを保存
        results_df_sorted = results_df.sort_values('AIC').head(10)
        results_df_sorted.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        print(f"\n上位10モデルを保存: model_comparison.csv")
        
        return self.best_params
    
    def fit_model(self, order=(1,1,1), seasonal_order=(1,1,1,7), use_exog=True):
        """
        指定されたパラメータでSARIMAXモデルを訓練
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) 非季節ARIMAパラメータ
        seasonal_order : tuple
            (P, D, Q, m) 季節ARIMAパラメータ
        use_exog : bool
            外生変数を使用するか
        """
        print(f"\n【SARIMAXモデル訓練】")
        print("-" * 80)
        print(f"パラメータ: SARIMA{order}x{seasonal_order}")
        print(f"外生変数使用: {use_exog}")
        
        # 外生変数の準備
        if use_exog:
            exog_cols = ['is_weekend', 'is_month_start', 'is_month_end'] + \
                       [f'dow_{i}' for i in range(7)] + \
                       [f'month_{i}' for i in range(1, 13)]
            exog_train = self.train[exog_cols]
        else:
            exog_train = None
        
        # モデル構築
        self.model = SARIMAX(self.train['y'],
                            order=order,
                            seasonal_order=seasonal_order,
                            exog=exog_train,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        
        # モデル訓練
        print("\n訓練中...")
        self.model_fit = self.model.fit(disp=False, maxiter=200)
        
        self.best_params = {
            'order': order,
            'seasonal_order': seasonal_order,
            'use_exog': use_exog
        }
        
        print("訓練完了!")
        print(f"AIC: {self.model_fit.aic:.2f}")
        print(f"BIC: {self.model_fit.bic:.2f}")
        print("-" * 80)
        
        return self.model_fit
    
    def predict(self, steps=None):
        """
        予測を実行
        
        Parameters:
        -----------
        steps : int or None
            予測するステップ数。Noneの場合はテストデータサイズ
        """
        if self.model_fit is None:
            raise ValueError("先にモデルを訓練してください")
        
        if steps is None:
            steps = len(self.test)
        
        # 外生変数の準備
        if self.best_params['use_exog']:
            exog_cols = ['is_weekend', 'is_month_start', 'is_month_end'] + \
                       [f'dow_{i}' for i in range(7)] + \
                       [f'month_{i}' for i in range(1, 13)]
            exog_test = self.test[exog_cols].iloc[:steps]
        else:
            exog_test = None
        
        # 予測実行
        forecast = self.model_fit.forecast(steps=steps, exog=exog_test)
        
        return forecast
    
    def evaluate(self, output_dir='./sarimax_output'):
        """モデルの評価とメトリクス計算"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【モデル評価】")
        print("-" * 80)
        
        # 予測実行
        forecast = self.predict()
        
        # メトリクス計算
        y_true = self.test['y'].values
        y_pred = forecast.values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # 訓練データの統計
        train_mean = self.train['y'].mean()
        train_std = self.train['y'].std()
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse_val,
            'MAPE': mape,
            'R2': r2,
            'MAE/Mean': mae / train_mean,
            'RMSE/Std': rmse_val / train_std
        }
        
        print("評価メトリクス:")
        print(f"  MAE (Mean Absolute Error): {mae:.2f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse_val:.2f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE/訓練データ平均: {metrics['MAE/Mean']:.2%}")
        print(f"  RMSE/訓練データ標準偏差: {metrics['RMSE/Std']:.2%}")
        
        # メトリクスを保存
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{output_dir}/evaluation_metrics.csv', index=False)
        print(f"\nメトリクスを保存: evaluation_metrics.csv")
        
        return metrics, forecast
    
    def plot_forecast(self, forecast, output_dir='./sarimax_output'):
        """予測結果をプロット"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【予測結果のプロット】")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 全体像
        axes[0].plot(self.train.index, self.train['y'], label='Training Data', linewidth=1)
        axes[0].plot(self.test.index, self.test['y'], label='Actual Test Data', 
                    linewidth=1.5, color='green')
        axes[0].plot(self.test.index[:len(forecast)], forecast, 
                    label='Forecast', linewidth=1.5, color='red', linestyle='--')
        axes[0].axvline(x=self.test.index[0], color='black', linestyle=':', alpha=0.5)
        axes[0].set_title('SARIMAX Forecast - Full View', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Call Volume')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # テスト期間のズームイン
        axes[1].plot(self.test.index, self.test['y'], label='Actual', 
                    linewidth=2, marker='o', markersize=4, color='green')
        axes[1].plot(self.test.index[:len(forecast)], forecast, 
                    label='Forecast', linewidth=2, marker='s', markersize=4, 
                    color='red', linestyle='--')
        axes[1].set_title('SARIMAX Forecast - Test Period Detail', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Call Volume')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # 残差プロット
        residuals = self.test['y'].values[:len(forecast)] - forecast.values
        axes[2].plot(self.test.index[:len(forecast)], residuals, 
                    linewidth=1.5, marker='o', markersize=4, color='purple')
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2].fill_between(self.test.index[:len(forecast)], residuals, 0, alpha=0.3)
        axes[2].set_title('Forecast Residuals', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Residual (Actual - Forecast)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/forecast_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: forecast_results.png")
    
    def plot_diagnostics(self, output_dir='./sarimax_output'):
        """モデル診断プロット"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n【モデル診断】")
        
        fig = plt.figure(figsize=(15, 12))
        self.model_fit.plot_diagnostics(fig=fig)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: model_diagnostics.png")
    
    def forecast_future(self, days=30, output_dir='./sarimax_output'):
        """
        将来の予測を実行
        
        Parameters:
        -----------
        days : int
            予測する日数
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n【将来予測 - {days}日間】")
        print("-" * 80)
        
        # 全データで再訓練
        if self.best_params['use_exog']:
            exog_cols = ['is_weekend', 'is_month_start', 'is_month_end'] + \
                       [f'dow_{i}' for i in range(7)] + \
                       [f'month_{i}' for i in range(1, 13)]
            exog_full = self.df[exog_cols]
        else:
            exog_full = None
        
        model_full = SARIMAX(self.df['y'],
                            order=self.best_params['order'],
                            seasonal_order=self.best_params['seasonal_order'],
                            exog=exog_full,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        
        model_fit_full = model_full.fit(disp=False, maxiter=200)
        
        # 将来の日付を生成
        last_date = self.df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=days, freq='D')
        
        # 将来の外生変数を作成
        if self.best_params['use_exog']:
            future_exog = pd.DataFrame(index=future_dates)
            future_exog['dayofweek'] = future_exog.index.dayofweek
            future_exog['month'] = future_exog.index.month
            future_exog['is_weekend'] = (future_exog['dayofweek'] >= 5).astype(int)
            future_exog['is_month_start'] = future_exog.index.is_month_start.astype(int)
            future_exog['is_month_end'] = future_exog.index.is_month_end.astype(int)
            
            for i in range(7):
                future_exog[f'dow_{i}'] = (future_exog['dayofweek'] == i).astype(int)
            
            for i in range(1, 13):
                future_exog[f'month_{i}'] = (future_exog['month'] == i).astype(int)
            
            future_exog = future_exog[exog_cols]
        else:
            future_exog = None
        
        # 予測実行（信頼区間付き）
        forecast_result = model_fit_full.get_forecast(steps=days, exog=future_exog)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95%信頼区間
        
        # 結果をDataFrameに
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_mean.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values
        })
        
        # 保存
        forecast_df.to_csv(f'{output_dir}/future_forecast_{days}days.csv', index=False)
        print(f"将来予測を保存: future_forecast_{days}days.csv")
        
        # プロット
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # 過去データ（最近90日）
        recent_data = self.df.tail(90)
        ax.plot(recent_data.index, recent_data['y'], label='Historical Data', 
               linewidth=1.5, color='blue')
        
        # 予測
        ax.plot(future_dates, forecast_mean, label='Forecast', 
               linewidth=2, color='red', linestyle='--')
        
        # 信頼区間
        ax.fill_between(future_dates, 
                       forecast_ci.iloc[:, 0], 
                       forecast_ci.iloc[:, 1],
                       alpha=0.3, color='red', label='95% Confidence Interval')
        
        ax.axvline(x=self.df.index[-1], color='black', linestyle=':', 
                  alpha=0.5, label='Forecast Start')
        
        ax.set_title(f'Future Forecast - {days} Days', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Call Volume')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/future_forecast_{days}days.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  プロット保存: future_forecast_{days}days.png")
        
        print("\n予測統計:")
        print(f"  平均予測値: {forecast_mean.mean():.2f}")
        print(f"  最小予測値: {forecast_mean.min():.2f}")
        print(f"  最大予測値: {forecast_mean.max():.2f}")
        print(f"  標準偏差: {forecast_mean.std():.2f}")
        print("-" * 80)
        
        return forecast_df
    
    def summary(self):
        """モデルのサマリーを表示"""
        if self.model_fit is None:
            raise ValueError("先にモデルを訓練してください")
        
        print("\n【モデルサマリー】")
        print("=" * 80)
        print(self.model_fit.summary())
        print("=" * 80)
    
    def run_full_pipeline(self, output_dir='./sarimax_output', 
                         auto_search=True, forecast_days=30):
        """
        完全なパイプラインを実行
        
        Parameters:
        -----------
        output_dir : str
            出力ディレクトリ
        auto_search : bool
            自動パラメータ探索を行うか
        forecast_days : int
            将来予測の日数
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("SARIMAX予測パイプライン開始")
        print("=" * 80)
        
        # 1. 定常性チェック
        self.check_stationarity()
        
        # 2. ACF/PACFプロット
        self.plot_acf_pacf(output_dir)
        
        # 3. モデル訓練
        if auto_search:
            self.auto_arima_search(output_dir=output_dir, max_iterations=30)
        else:
            self.fit_model(order=(1,1,1), seasonal_order=(1,1,1,7), use_exog=True)
        
        # 4. モデルサマリー
        self.summary()
        
        # 5. モデル診断
        self.plot_diagnostics(output_dir)
        
        # 6. 評価
        metrics, forecast = self.evaluate(output_dir)
        
        # 7. 予測プロット
        self.plot_forecast(forecast, output_dir)
        
        # 8. 将来予測
        forecast_df = self.forecast_future(days=forecast_days, output_dir=output_dir)
        
        print("\n" + "=" * 80)
        print("パイプライン完了!")
        print(f"全ての結果は '{output_dir}' に保存されています")
        print("=" * 80)
        
        return metrics, forecast_df


# =============================================================================
# メイン実行例
# =============================================================================

if __name__ == "__main__":
    """
    使用例:
    
    # CSVファイルからデータを読み込み
    df = pd.read_csv('your_data.csv')
    
    # SARIMAXモデルで予測
    sarimax = CallCenterSARIMAX(df, test_size=30)
    metrics, forecast_df = sarimax.run_full_pipeline(
        output_dir='./sarimax_results',
        auto_search=True,
        forecast_days=30
    )
    """
    
    print("\n" + "="*80)
    print("このスクリプトを使用するには:")
    print("="*80)
    print("""
    import pandas as pd
    from call_center_sarimax_forecast import CallCenterSARIMAX
    
    # データ読み込み
    df = pd.read_csv('your_data.csv')  # ds, y カラムを含むCSV
    
    # SARIMAXモデルで予測（完全自動）
    sarimax = CallCenterSARIMAX(df, test_size=30)
    metrics, forecast_df = sarimax.run_full_pipeline(
        output_dir='./sarimax_results',
        auto_search=True,      # 自動パラメータ探索
        forecast_days=30       # 30日先まで予測
    )
    
    # または手動でパラメータ指定
    sarimax = CallCenterSARIMAX(df, test_size=30)
    sarimax.fit_model(order=(1,1,1), seasonal_order=(1,1,1,7), use_exog=True)
    metrics, forecast = sarimax.evaluate()
    forecast_df = sarimax.forecast_future(days=30)
    """)
    print("="*80)
