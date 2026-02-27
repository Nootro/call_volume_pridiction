"""
コールセンター呼量データの外れ値診断ツール
==========================================
Prophet形式（ds, y）のデータに対して、外れ値の検出・診断・処理推奨を行う。

使い方:
    python outlier_diagnosis.py --input data.csv --output results/
    python outlier_diagnosis.py --input data.csv --threshold 3.0
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

try:
    from prophet import Prophet
except ImportError:
    print("prophetがインストールされていません: pip install prophet")
    sys.exit(1)


# ============================================================
# 1. データ読み込み・基本バリデーション
# ============================================================

def load_and_validate(filepath: str) -> pd.DataFrame:
    """Prophet形式(ds, y)のCSVを読み込み、基本的なバリデーションを行う。"""
    df = pd.read_csv(filepath)

    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("CSVには 'ds' と 'y' カラムが必要です。")

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.sort_values("ds").reset_index(drop=True)

    n_missing = df["y"].isna().sum()
    if n_missing > 0:
        print(f"[警告] yに {n_missing} 件の欠損値があります。外れ値診断では除外されます。")

    n_dup = df["ds"].duplicated().sum()
    if n_dup > 0:
        print(f"[警告] dsに {n_dup} 件の重複日付があります。最初の値を使用します。")
        df = df.drop_duplicates(subset="ds", keep="first").reset_index(drop=True)

    print(f"[INFO] データ期間: {df['ds'].min().date()} ~ {df['ds'].max().date()}")
    print(f"[INFO] レコード数: {len(df)}")
    print(f"[INFO] y の基本統計量:")
    print(f"         mean={df['y'].mean():.1f}, std={df['y'].std():.1f}")
    print(f"         min={df['y'].min():.1f}, max={df['y'].max():.1f}")
    print()

    return df


# ============================================================
# 2. 複数手法による外れ値検出
# ============================================================

def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Z-score法による外れ値検出。"""
    z = np.abs(stats.zscore(df["y"].dropna()))
    flags = pd.Series(False, index=df.index)
    valid_idx = df["y"].dropna().index
    flags.loc[valid_idx] = z > threshold
    return flags


def detect_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    """IQR法による外れ値検出。"""
    q1 = df["y"].quantile(0.25)
    q3 = df["y"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (df["y"] < lower) | (df["y"] > upper)


def detect_iqr_by_dow(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    """曜日別IQR法。コールセンターでは曜日効果が大きいため、曜日ごとに判定する。"""
    df_work = df.copy()
    df_work["dow"] = df_work["ds"].dt.dayofweek
    flags = pd.Series(False, index=df.index)

    for dow in range(7):
        mask = df_work["dow"] == dow
        subset = df_work.loc[mask, "y"]
        if len(subset) < 10:
            continue
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        flags.loc[mask] = (subset < lower) | (subset > upper)

    return flags


def detect_rolling_zscore(
    df: pd.DataFrame, window: int = 28, threshold: float = 3.0
) -> pd.Series:
    """ローリングウィンドウZ-score法。局所的な外れ値を検出する。"""
    rolling_mean = df["y"].rolling(window=window, center=True, min_periods=7).mean()
    rolling_std = df["y"].rolling(window=window, center=True, min_periods=7).std()
    z = np.abs((df["y"] - rolling_mean) / rolling_std.replace(0, np.nan))
    return z > threshold


def detect_prophet_residual(
    df: pd.DataFrame, threshold: float = 3.0
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Prophetの残差ベースの外れ値検出。
    Prophetでフィット → 残差を計算 → 残差のZ-scoreで判定。
    季節性・トレンドを除去した上での外れ値を検出できる。
    """
    print("[INFO] Prophet残差法: モデルをフィット中...")
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.add_country_holidays(country_name="JP")

    df_valid = df.dropna(subset=["y"]).copy()
    m.fit(df_valid[["ds", "y"]])

    fitted = m.predict(df_valid[["ds"]])
    residuals = df_valid["y"].values - fitted["yhat"].values
    residual_z = np.abs(stats.zscore(residuals))

    flags = pd.Series(False, index=df.index)
    flags.loc[df_valid.index] = residual_z > threshold

    # 残差情報を返す（診断用）
    residual_df = pd.DataFrame(
        {
            "ds": df_valid["ds"].values,
            "y": df_valid["y"].values,
            "yhat": fitted["yhat"].values,
            "residual": residuals,
            "residual_z": residual_z,
        }
    )

    print(f"[INFO] Prophet残差法: 完了（残差std={np.std(residuals):.1f}）")
    return flags, residual_df


# ============================================================
# 3. 外れ値診断の統合
# ============================================================

def run_all_detections(
    df: pd.DataFrame, z_threshold: float = 3.0, iqr_factor: float = 1.5
) -> pd.DataFrame:
    """全手法を実行し、結果を統合する。"""
    print("=" * 60)
    print("外れ値検出を実行中...")
    print("=" * 60)
    print()

    result = df[["ds", "y"]].copy()

    # 各手法の実行
    result["outlier_zscore"] = detect_zscore(df, threshold=z_threshold)
    result["outlier_iqr"] = detect_iqr(df, factor=iqr_factor)
    result["outlier_iqr_dow"] = detect_iqr_by_dow(df, factor=iqr_factor)
    result["outlier_rolling_z"] = detect_rolling_zscore(
        df, window=28, threshold=z_threshold
    )

    prophet_flags, residual_df = detect_prophet_residual(df, threshold=z_threshold)
    result["outlier_prophet"] = prophet_flags

    # 検出手法の合計数
    detection_cols = [
        "outlier_zscore",
        "outlier_iqr",
        "outlier_iqr_dow",
        "outlier_rolling_z",
        "outlier_prophet",
    ]
    result["n_methods_flagged"] = result[detection_cols].sum(axis=1)

    # カレンダー情報を付与
    result["dayofweek"] = result["ds"].dt.day_name()
    result["month"] = result["ds"].dt.month
    result["day"] = result["ds"].dt.day
    result["is_monthstart"] = result["ds"].dt.is_month_start
    result["is_monthend"] = result["ds"].dt.is_month_end

    return result, residual_df


# ============================================================
# 4. 診断レポート生成
# ============================================================

def print_outlier_list(result: pd.DataFrame, min_methods: int = 2) -> pd.DataFrame:
    """外れ値として検出された日付の一覧を表示。"""
    outliers = result[result["n_methods_flagged"] >= min_methods].copy()
    outliers = outliers.sort_values("n_methods_flagged", ascending=False)

    print()
    print("=" * 80)
    print(f"外れ値候補一覧（{min_methods}手法以上で検出: {len(outliers)} 件）")
    print("=" * 80)

    if len(outliers) == 0:
        print("  → 外れ値候補はありません。")
        return outliers

    display_cols = [
        "ds", "y", "dayofweek", "n_methods_flagged",
        "outlier_zscore", "outlier_iqr", "outlier_iqr_dow",
        "outlier_rolling_z", "outlier_prophet",
    ]
    print(outliers[display_cols].to_string(index=False))
    print()

    return outliers


def diagnose_outlier_impact(df: pd.DataFrame, result: pd.DataFrame) -> dict:
    """外れ値処理の必要性を診断する。"""
    diagnostics = {}

    total = len(df)
    n_any = (result["n_methods_flagged"] >= 1).sum()
    n_strong = (result["n_methods_flagged"] >= 3).sum()
    n_consensus = (result["n_methods_flagged"] >= 4).sum()

    diagnostics["total_records"] = total
    diagnostics["outliers_any_method"] = n_any
    diagnostics["outliers_3plus_methods"] = n_strong
    diagnostics["outliers_4plus_methods"] = n_consensus
    diagnostics["outlier_ratio_pct"] = round(n_strong / total * 100, 2)

    # 外れ値の曜日分布
    strong_outliers = result[result["n_methods_flagged"] >= 3]
    if len(strong_outliers) > 0:
        dow_dist = strong_outliers["dayofweek"].value_counts().to_dict()
    else:
        dow_dist = {}
    diagnostics["outlier_dow_distribution"] = dow_dist

    # 外れ値の月分布
    if len(strong_outliers) > 0:
        month_dist = strong_outliers["month"].value_counts().sort_index().to_dict()
    else:
        month_dist = {}
    diagnostics["outlier_month_distribution"] = month_dist

    # 外れ値の年分布
    if len(strong_outliers) > 0:
        year_dist = (
            strong_outliers["ds"]
            .dt.year.value_counts()
            .sort_index()
            .to_dict()
        )
    else:
        year_dist = {}
    diagnostics["outlier_year_distribution"] = year_dist

    # 連続外れ値の検出（2日以上連続は「イベント」の可能性）
    consecutive_groups = []
    if len(strong_outliers) > 0:
        sorted_dates = strong_outliers["ds"].sort_values().reset_index(drop=True)
        group_start = sorted_dates.iloc[0]
        group_end = sorted_dates.iloc[0]
        for i in range(1, len(sorted_dates)):
            if (sorted_dates.iloc[i] - sorted_dates.iloc[i - 1]).days <= 2:
                group_end = sorted_dates.iloc[i]
            else:
                if group_start != group_end:
                    consecutive_groups.append(
                        (group_start.date(), group_end.date())
                    )
                group_start = sorted_dates.iloc[i]
                group_end = sorted_dates.iloc[i]
        if group_start != group_end:
            consecutive_groups.append((group_start.date(), group_end.date()))
    diagnostics["consecutive_outlier_periods"] = consecutive_groups

    # 正規性検定（Shapiro-Wilk、サンプルが多すぎる場合はサブサンプル）
    y_valid = df["y"].dropna()
    if len(y_valid) > 5000:
        y_sample = y_valid.sample(5000, random_state=42)
    else:
        y_sample = y_valid
    _, p_value = stats.shapiro(y_sample)
    diagnostics["shapiro_p_value"] = round(p_value, 6)

    # 尖度・歪度
    diagnostics["skewness"] = round(y_valid.skew(), 3)
    diagnostics["kurtosis"] = round(y_valid.kurtosis(), 3)

    return diagnostics


def print_diagnosis_report(diagnostics: dict):
    """診断結果をレポートとして表示。"""
    print()
    print("=" * 80)
    print("外れ値診断レポート")
    print("=" * 80)
    print()

    print("■ 検出サマリー")
    print(f"  総レコード数:                  {diagnostics['total_records']}")
    print(f"  いずれかの手法で検出:          {diagnostics['outliers_any_method']} 件")
    print(f"  3手法以上で検出（強い外れ値）: {diagnostics['outliers_3plus_methods']} 件")
    print(f"  4手法以上で検出（確実な外れ値）:{diagnostics['outliers_4plus_methods']} 件")
    print(f"  外れ値比率（3手法以上）:       {diagnostics['outlier_ratio_pct']}%")
    print()

    print("■ 分布特性")
    print(f"  歪度 (skewness):  {diagnostics['skewness']}")
    print(f"  尖度 (kurtosis):  {diagnostics['kurtosis']}")
    print(f"  Shapiro-Wilk p値: {diagnostics['shapiro_p_value']}")
    if diagnostics["shapiro_p_value"] < 0.05:
        print("  → 正規分布から有意に逸脱しています（Z-score法の信頼性に注意）")
    else:
        print("  → 正規分布からの逸脱は有意ではありません")
    print()

    print("■ 外れ値の曜日分布（3手法以上）")
    if diagnostics["outlier_dow_distribution"]:
        for dow, cnt in diagnostics["outlier_dow_distribution"].items():
            print(f"    {dow}: {cnt} 件")
    else:
        print("    該当なし")
    print()

    print("■ 外れ値の月分布（3手法以上）")
    if diagnostics["outlier_month_distribution"]:
        for month, cnt in diagnostics["outlier_month_distribution"].items():
            print(f"    {month}月: {cnt} 件")
    else:
        print("    該当なし")
    print()

    print("■ 外れ値の年分布（3手法以上）")
    if diagnostics["outlier_year_distribution"]:
        for year, cnt in diagnostics["outlier_year_distribution"].items():
            print(f"    {year}年: {cnt} 件")
    else:
        print("    該当なし")
    print()

    print("■ 連続外れ値期間（2日以上連続、イベント等の可能性）")
    if diagnostics["consecutive_outlier_periods"]:
        for start, end in diagnostics["consecutive_outlier_periods"]:
            print(f"    {start} ~ {end}")
    else:
        print("    該当なし")
    print()

    # === 総合判断 ===
    print("■ 総合判断・推奨アクション")
    print("-" * 60)

    ratio = diagnostics["outlier_ratio_pct"]
    n_strong = diagnostics["outliers_3plus_methods"]
    n_consecutive = len(diagnostics["consecutive_outlier_periods"])
    kurtosis = diagnostics["kurtosis"]

    if n_strong == 0:
        print("  外れ値は検出されませんでした。外れ値処理は不要と判断します。")
        print("  → Prophetにそのままデータを渡して問題ありません。")
    elif ratio < 1.0 and n_consecutive == 0:
        print("  外れ値はごく少数で、散発的です。")
        print("  → 個別に確認し、明らかな異常（システム障害、データ入力ミス等）のみ")
        print("    処理することを推奨します。")
        print("  → Prophetの場合、該当日のyをNaNにするのが最もシンプルです。")
    elif ratio < 3.0:
        print("  一定数の外れ値が検出されました。")
        if n_consecutive > 0:
            print(f"  連続外れ値が {n_consecutive} 期間あります。")
            print("  → これらは「外れ値」ではなく「イベント期間」である可能性があります。")
            print("  → Prophetのholiday/eventとして明示的にモデル化することを推奨します。")
        print("  → 散発的な外れ値は yをNaN に置換して処理してください。")
        print("  → CatBoost/LightGBMには外れ値フラグを特徴量として渡す手もあります。")
    else:
        print(f"  外れ値比率が {ratio}% と高めです。")
        if kurtosis > 3:
            print(f"  尖度が {kurtosis:.1f} と高く、分布の裾が重い傾向があります。")
            print("  → これは「外れ値が多い」のではなく、データの分布自体が")
            print("    正規分布と異なる可能性があります。")
            print("  → 対数変換やBox-Cox変換を検討してください。")
            print("  → Prophetの場合、yを対数変換してからフィットし、")
            print("    予測値を逆変換する方法が有効です。")
        else:
            print("  → 閾値が厳しすぎる可能性があります。")
            print("  → --threshold を 3.5 や 4.0 に上げて再実行を検討してください。")

    print()


# ============================================================
# 5. 可視化
# ============================================================

def plot_outlier_overview(
    result: pd.DataFrame, output_dir: Path, min_methods: int = 3
):
    """外れ値の全体像を可視化する。"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle("Outlier Diagnosis Report", fontsize=16, fontweight="bold", y=0.98)

    outlier_mask = result["n_methods_flagged"] >= min_methods

    # --- Panel 1: 時系列 + 外れ値ハイライト ---
    ax = axes[0]
    ax.plot(result["ds"], result["y"], color="black", linewidth=0.7, alpha=0.8, label="y")
    if outlier_mask.any():
        ax.scatter(
            result.loc[outlier_mask, "ds"],
            result.loc[outlier_mask, "y"],
            color="red", s=40, zorder=5, label=f"Outlier (≥{min_methods} methods)",
        )
    ax.set_title("Time Series with Outliers Highlighted")
    ax.set_ylabel("y (Call Volume)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Panel 2: 検出手法数のヒートマップ（日別） ---
    ax = axes[1]
    scatter = ax.scatter(
        result["ds"],
        result["y"],
        c=result["n_methods_flagged"],
        cmap="YlOrRd",
        s=15,
        vmin=0,
        vmax=5,
    )
    plt.colorbar(scatter, ax=ax, label="Number of Methods Flagged")
    ax.set_title("Detection Consensus Heatmap")
    ax.set_ylabel("y (Call Volume)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Panel 3: 曜日別箱ひげ図 + 外れ値 ---
    ax = axes[2]
    result_work = result.copy()
    dow_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    result_work["dow_num"] = result_work["ds"].dt.dayofweek
    bp_data = [
        result_work.loc[result_work["dow_num"] == i, "y"].dropna().values
        for i in range(7)
    ]
    bp = ax.boxplot(bp_data, labels=dow_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    if outlier_mask.any():
        outlier_rows = result_work.loc[outlier_mask]
        ax.scatter(
            outlier_rows["dow_num"] + 1,
            outlier_rows["y"],
            color="red", s=30, zorder=5, label="Outlier",
        )
    ax.set_title("Distribution by Day of Week")
    ax.set_ylabel("y (Call Volume)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: 月別箱ひげ図 + 外れ値 ---
    ax = axes[3]
    bp_data_month = [
        result.loc[result["month"] == m, "y"].dropna().values for m in range(1, 13)
    ]
    month_labels = [f"{m}月" for m in range(1, 13)]
    bp2 = ax.boxplot(bp_data_month, labels=month_labels, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("lightyellow")
        patch.set_alpha(0.7)

    if outlier_mask.any():
        outlier_rows = result.loc[outlier_mask]
        ax.scatter(
            outlier_rows["month"],
            outlier_rows["y"],
            color="red", s=30, zorder=5, label="Outlier",
        )
    ax.set_title("Distribution by Month")
    ax.set_ylabel("y (Call Volume)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / "outlier_overview.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] {save_path}")


def plot_prophet_residuals(residual_df: pd.DataFrame, output_dir: Path):
    """Prophet残差の可視化。"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("Prophet Residual Analysis", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: 実測 vs 予測 ---
    ax = axes[0]
    ax.plot(residual_df["ds"], residual_df["y"], color="black", linewidth=0.7, label="Actual")
    ax.plot(residual_df["ds"], residual_df["yhat"], color="blue", linewidth=0.7, alpha=0.7, label="Prophet Fitted")
    ax.set_title("Actual vs Prophet Fitted")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: 残差の時系列 ---
    ax = axes[1]
    colors = ["red" if z > 3 else "gray" for z in residual_df["residual_z"]]
    ax.scatter(residual_df["ds"], residual_df["residual"], c=colors, s=10, alpha=0.6)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Residuals Over Time (Red = |Z| > 3)")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: 残差のヒストグラム ---
    ax = axes[2]
    ax.hist(residual_df["residual"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / "prophet_residuals.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] {save_path}")


# ============================================================
# 6. 外れ値リストのCSVエクスポート
# ============================================================

def export_results(result: pd.DataFrame, output_dir: Path):
    """全データ + 外れ値フラグをCSVで出力。"""
    # 全データ（フラグ付き）
    full_path = output_dir / "outlier_flags_full.csv"
    result.to_csv(full_path, index=False)
    print(f"[保存] {full_path}")

    # 外れ値のみ抽出（1手法以上）
    outliers_any = result[result["n_methods_flagged"] >= 1].copy()
    outlier_path = output_dir / "outlier_dates.csv"
    export_cols = [
        "ds", "y", "dayofweek", "month", "day",
        "is_monthstart", "is_monthend", "n_methods_flagged",
        "outlier_zscore", "outlier_iqr", "outlier_iqr_dow",
        "outlier_rolling_z", "outlier_prophet",
    ]
    outliers_any[export_cols].to_csv(outlier_path, index=False)
    print(f"[保存] {outlier_path}")

    # Prophet用のクレンジング済みデータ（3手法以上をNaN化）
    cleaned = result[["ds", "y"]].copy()
    strong_mask = result["n_methods_flagged"] >= 3
    cleaned.loc[strong_mask, "y"] = np.nan
    cleaned_path = output_dir / "cleaned_for_prophet.csv"
    cleaned.to_csv(cleaned_path, index=False)
    n_cleaned = strong_mask.sum()
    print(f"[保存] {cleaned_path} （{n_cleaned} 件をNaN化）")


# ============================================================
# 7. メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="コールセンター呼量データの外れ値診断ツール"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="入力CSVファイルパス（ds, y形式）"
    )
    parser.add_argument(
        "--output", "-o", default="outlier_results",
        help="出力ディレクトリ（デフォルト: outlier_results/）",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=3.0,
        help="Z-score閾値（デフォルト: 3.0）",
    )
    parser.add_argument(
        "--iqr-factor", type=float, default=1.5,
        help="IQR倍率（デフォルト: 1.5）",
    )
    parser.add_argument(
        "--min-methods", type=int, default=3,
        help="外れ値と判定する最小検出手法数（デフォルト: 3）",
    )
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 実行
    print()
    print("=" * 80)
    print("  コールセンター呼量データ 外れ値診断ツール")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Step 1: データ読み込み
    df = load_and_validate(args.input)

    # Step 2: 外れ値検出
    result, residual_df = run_all_detections(
        df, z_threshold=args.threshold, iqr_factor=args.iqr_factor
    )

    # Step 3: 外れ値一覧表示
    print_outlier_list(result, min_methods=args.min_methods)

    # Step 4: 診断
    diagnostics = diagnose_outlier_impact(df, result)
    print_diagnosis_report(diagnostics)

    # Step 5: 可視化
    print("-" * 60)
    print("可視化を生成中...")
    plot_outlier_overview(result, output_dir, min_methods=args.min_methods)
    plot_prophet_residuals(residual_df, output_dir)

    # Step 6: CSV出力
    print()
    print("-" * 60)
    print("結果をCSV出力中...")
    export_results(result, output_dir)

    print()
    print("=" * 80)
    print("  完了！")
    print(f"  結果は {output_dir}/ に保存されています。")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
