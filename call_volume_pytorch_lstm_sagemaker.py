#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改良版の PyTorch LSTM モデルによるマルチステップ呼量予測。

このスクリプトは、ベースラインの LSTM 実装を拡張し、日次の呼量データから
複数日の先行予測を行うためのものです。呼量予測には季節性や週次パターン、
祝日効果などが影響するため、日時データから派生するカレンダー特徴量や
周期的なサイン・コサインエンコーディング、祝日フラグなどを追加します。
また、LSTM によるシーケンス学習だけでなく、入力シーケンスの長さや隠れ層
のサイズ、層数などを調整できるようにしています。

実装の流れは以下の通りです。

1. **データ読み込みと特徴量生成**
   `ds` 列（日付）と `y` 列（呼量）を持つ CSV ファイルを読み込みます。
   日付から曜日、月、日、年、週番号などのカレンダー属性を作成し、
   `sin`/`cos` による周期エンコードを加えます。日本の祝日ライブラリ
   (`holidays` や `jpholiday`) が利用可能な場合は祝日フラグを立てます。
   これらの特徴量を `y` と結合した多変量時系列としてモデルに渡します。

2. **訓練／検証分割**
   データを時系列順に分割し、最後の `horizon` 日間を検証データとします。
   これにより未来情報のリークを防ぎ、現実的な予測性能を評価できます。

3. **スライディングウィンドウ作成**
   過去 `seq_length` 日のデータを 1 つの入力シーケンスとし、
   その直後の `horizon` 日分の `y` を出力ターゲットとするサンプルを
   データセットとして作成します。これにより、モデルは一度の推論で
   複数日の予測を行うことが可能になります。

4. **標準化**
   入力特徴量は `StandardScaler` を用いて平均 0、分散 1 に正規化します。
   スケーラーは訓練データに対して学習し、同じ変換を検証データにも適用します。

5. **モデル構築**
   LSTM 層と全結合層からなるニューラルネットワークを定義します。
   LSTM 層ではバッチサイズ・シーケンス長・特徴量数を受け取り、最後の
   隠れ状態を全結合層に渡して `horizon` 次元の出力ベクトルに変換します。
   層数や隠れユニット数、ドロップアウト率はハイパーパラメータとして調整可能です。

6. **訓練ループ**
   平均二乗誤差 (MSE) を損失関数として採用し、`Adam` オプティマイザで
   重みを更新します。訓練は指定したエポック数だけ繰り返され、
   進捗としてエポックごとの損失を表示します。

7. **評価と予測**
   学習後、検証データに対する RMSE を計算し、モデル性能を報告します。
   また、最新の `seq_length` 日を入力として使い、将来 `horizon` 日間の
   予測値を生成します。予測結果は日付とともに表示されます。

SageMaker などの環境で利用しやすいよう、このスクリプトはコマンドライン
引数を使用しません。`main()` 内の変数 `data_file`、`seq_length`、`horizon`、
`hidden_size`、`num_layers`、`epochs`、`batch_size`、`learning_rate`、
`use_gpu` を変更することで設定を調整してください。
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def generate_jp_holidays(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """start から end まで（両端を含む）の日本の祝日の集合を返します。"""
    holidays: set[pd.Timestamp] = set()
    try:
        import holidays as pyholidays  # type: ignore[import]
        jp_holidays = pyholidays.country_holidays(
            "JP", years=list(range(start.year, end.year + 1))
        )
        holidays = {
            pd.Timestamp(day)
            for day in jp_holidays.keys()
            if start <= pd.Timestamp(day) <= end
        }
    except Exception:
        try:
            import jpholiday  # type: ignore[import]

            for date in pd.date_range(start, end, freq="D"):
                if jpholiday.is_holiday(date.to_pydatetime()):
                    holidays.add(date)
        except Exception:
            logging.warning(
                "`holidays` または `jpholiday` をインポートできませんでした。祝日フラグは 0 になります。"
            )
            holidays = set()
    return holidays


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame にカレンダーおよび周期的な特徴量を追加します。"""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day_of_month"] = df["ds"].dt.day
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["ds"].dt.quarter
    df["year"] = df["ds"].dt.year
    # 周期エンコーディング
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    # 祝日フラグ
    holidays = generate_jp_holidays(df["ds"].min(), df["ds"].max())
    df["is_holiday"] = df["ds"].isin(holidays).astype(int)
    return df


def create_dataset(
    series: pd.DataFrame,
    seq_length: int,
    horizon: int,
    feature_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """多変量時系列を入力シーケンスと出力シーケンスに変換します。

    パラメータ
    ----------
    series : pd.DataFrame
        少なくとも目的列 'y' とその他の特徴量を含むデータフレーム。
    seq_length : int
        入力として使用する過去の観測数。
    horizon : int
        予測する将来ポイントの数。
    feature_columns : list of str
        'y' を含む、特徴量として使用する列名のリスト。

    戻り値
    -------
    X : np.ndarray
        形状 (n_samples, seq_length, num_features) の配列。
    y : np.ndarray
        将来のターゲットを含む形状 (n_samples, horizon) の配列。
    """
    data = series[feature_columns].values
    target = series["y"].values
    X: List[np.ndarray] = []
    y_out: List[np.ndarray] = []
    total_length = len(series)
    for i in range(total_length - seq_length - horizon + 1):
        X.append(data[i : i + seq_length, :])
        y_out.append(target[i + seq_length : i + seq_length + horizon])
    return np.array(X), np.array(y_out)


class LSTMForecaster(nn.Module):
    """LSTM を用いたマルチステップ予測モデル。"""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播計算。

        パラメータ
        ----------
        x : torch.Tensor
            入力テンソル。形状は (バッチサイズ, シーケンス長, 特徴量数)。

        戻り値
        -------
        torch.Tensor
            予測結果。形状は (バッチサイズ, horizon)。
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (層数, バッチサイズ, 隠れユニット数)
        out = h_n[-1]  # 最後の層の隠れ状態を使用
        return self.fc(out)


def main() -> None:
    """スクリプトのエントリポイント。パラメータを直接編集して実行します。"""
    # === 設定値（必要に応じて変更してください） ===
    data_file: str | None = None  # 'ds' と 'y' を含む CSV ファイルへのパス。None の場合は合成データを生成。
    seq_length: int = 90          # 過去何日分のデータを入力とするか
    horizon: int = 60             # 予測する日数
    hidden_size: int = 64         # LSTM の隠れ層ユニット数
    num_layers: int = 1           # LSTM の層数
    epochs: int = 50              # 学習エポック数
    batch_size: int = 32          # バッチサイズ
    learning_rate: float = 1e-3   # 学習率
    use_gpu: bool = False         # True にすると GPU (cuda) を使用（利用可能な場合）

    # === データの読み込みまたは生成 ===
    if data_file:
        df_raw = pd.read_csv(data_file)
        if "ds" not in df_raw.columns or "y" not in df_raw.columns:
            raise ValueError("入力 CSV には 'ds' と 'y' 列が必要です。")
    else:
        # デモ用に LightGBM スクリプトと同様の合成データを生成
        print("data_file が指定されていません。合成データを生成します。")
        n_days = 3 * 365
        date_range = pd.date_range(
            start=pd.Timestamp.today() - pd.Timedelta(days=n_days),
            periods=n_days,
            freq="D",
        )
        trend = np.linspace(100, 200, n_days)
        weekly = 20 * np.sin(2 * np.pi * date_range.dayofweek / 7.0)
        annual = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.0)
        noise = np.random.normal(scale=5, size=n_days)
        y = trend + weekly + annual + noise
        df_raw = pd.DataFrame({"ds": date_range, "y": y})

    # カレンダー特徴量を追加
    df_features = add_calendar_features(df_raw)
    # 特徴量列のリスト（最初にターゲット）
    feature_cols: List[str] = [
        "y",
        "day_of_week",
        "month",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "quarter",
        "year",
        "sin_doy",
        "cos_doy",
        "sin_dow",
        "cos_dow",
        "is_holiday",
    ]
    # 祝日や他の特徴による NaN を含む行を削除
    df_features = df_features.dropna().reset_index(drop=True)

    # 訓練/検証の分割：最後の horizon 日を検証用に使用
    train_df = df_features.iloc[:-horizon]
    # 検証データセット作成時にシーケンス長分の履歴を含める
    valid_df = df_features.iloc[-(horizon + seq_length) :]

    # データセットを作成
    X_train, y_train = create_dataset(train_df, seq_length, horizon, feature_cols)
    X_valid, y_valid = create_dataset(valid_df, seq_length, horizon, feature_cols)
    # 特徴量を標準化
    scaler = StandardScaler()
    # スケーラー用にリシェイプ: (n_samples * seq_length, num_features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_valid_scaled = scaler.transform(X_valid.reshape(-1, X_valid.shape[2])).reshape(X_valid.shape)

    # テンソルに変換
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32, device=device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32, device=device)

    # DataLoader を作成
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # モデルを初期化
    model = LSTMForecaster(
        num_features=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        dropout=0.2 if num_layers > 1 else 0.0,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練ループ
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        # 経過を定期的に表示
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Training Loss: {epoch_loss:.4f}")

    # 評価
    model.eval()
    with torch.no_grad():
        preds_valid = model(X_valid_tensor)
    # ホライズン全体の RMSE を計算
    rmse = np.sqrt(
        mean_squared_error(
            y_valid_tensor.cpu().numpy().flatten(),
            preds_valid.cpu().numpy().flatten(),
        )
    )
    print(f"Validation RMSE (flattened over horizon): {rmse:.4f}")

    # 最終予測：完全なデータセット（訓練 + 検証）の最後の seq_length 日を使用
    full_df = df_features.copy()
    last_window = full_df.iloc[-seq_length:]
    # 最後のウィンドウの特徴行列を構築
    X_last = last_window[feature_cols].values
    X_last_scaled = scaler.transform(X_last)
    X_last_scaled = torch.tensor(
        X_last_scaled[np.newaxis, :, :], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        future_pred = model(X_last_scaled).cpu().numpy().flatten()
    # 将来の日付と予測値を持つ DataFrame を作成
    last_date = full_df["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecast_df = pd.DataFrame({"ds": future_dates, "y_pred": future_pred})
    print(f"\nMulti-step forecast (next {horizon} days):")
    print(forecast_df.head())


if __name__ == "__main__":
    main()