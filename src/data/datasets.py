import glob
import os

import numpy as np
import pandas as pd

from .transforms import create_sequences


class StockDataset:
    def __init__(
        self,
        params: dict,
        input_seq_length: int = 50,
        output_seq_length: int = 1,
    ) -> None:
        self.params = params
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.train_dataset = None
        self.test_dataset = None
        self.load_data()

    def __len__(self):
        return len(self.train_dataset[0])

    def __getitem__(self, idx):
        return self.train_dataset[0][idx], self.train_dataset[1][idx]

    def load_data(self, display_summary: bool = False) -> None:
        try:
            import joblib
            import torch
            from sklearn.preprocessing import MinMaxScaler
        except ImportError as exc:
            raise ImportError(
                "torch, joblib, and scikit-learn are required for StockDataset"
            ) from exc

        params = self.params
        scaler_params = params.get("scaler", {})
        scaler_path = scaler_params.get("path", "scaler.pkl")

        if scaler_params.get("load") and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))

        data_path = params["data"]["path"]
        df = pd.read_csv(data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            df = df.drop(columns=["Date"])

        if scaler_params.get("fit", True):
            data_scaled = scaler.fit_transform(df)
            joblib.dump(scaler, scaler_path)
        else:
            data_scaled = scaler.transform(df)

        n_examples, n_features = df.shape
        x_values, y_values = create_sequences(
            data_scaled,
            output_idx=0,
            seq_length=self.input_seq_length,
        )
        n_batches = x_values.shape[0]

        x_tensor = torch.tensor(
            x_values.reshape(n_batches, self.input_seq_length, n_features),
            dtype=torch.float32,
        )
        y_tensor = torch.tensor(
            y_values.reshape(n_batches, self.output_seq_length),
            dtype=torch.float32,
        )

        train_size = int(len(x_tensor) * 0.8)
        self.train_dataset = [x_tensor[:train_size], y_tensor[:train_size]]
        self.test_dataset = [x_tensor[train_size:], y_tensor[train_size:]]

        if display_summary:
            print(f"Examples: {n_examples}")
            print(f"Features: {n_features}")
            print(f"Batches: {n_batches}")


class DatasetGenerator:
    def __init__(
        self,
        data_glob: str = "data/*_data.csv",
        scaler_dir: str = "scalers",
        load_scaler: bool = False,
    ) -> None:
        self.data_glob = data_glob
        self.scaler_dir = scaler_dir
        self.load_scaler = load_scaler
        self.data_params = self.create_dataset()
        self.items = self.data_params.keys()

    def create_dataset(self) -> dict:
        params = {}

        for ticker_file in glob.glob(self.data_glob):
            ticker = os.path.basename(ticker_file).replace(".NS_data.csv", "")
            scaler_path = os.path.join(self.scaler_dir, f"{ticker}_scaler.pkl")
            params[ticker] = {
                "data": {"path": ticker_file, "ticker": ticker},
                "scaler": {
                    "path": scaler_path,
                    "load": self.load_scaler and os.path.exists(scaler_path),
                    "fit": not (self.load_scaler and os.path.exists(scaler_path)),
                },
            }

        return params

    def get_dataset(self, ticker: str) -> StockDataset:
        return StockDataset(self.data_params[ticker])


class StockDataLoader:
    pass
