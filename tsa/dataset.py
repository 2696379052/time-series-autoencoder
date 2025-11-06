import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pkg_resources
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from export.lobster_toolkit.core.builder import LOBSTERDataBuilder

try:  # Hydra may not be installed in minimal environments
    from hydra.utils import to_absolute_path as _to_absolute_path
except ImportError:  # pragma: no cover
    _to_absolute_path = None


class Tasks(Enum):
    prediction = "prediction"
    reconstruction = "reconstruction"


class TimeSeriesDataset(object):
    def __init__(self, task: Tasks, data_path: str, categorical_cols: List[str], index_col: str, target_col: str,
                 seq_length: int, batch_size: int, prediction_window: int = 1):
        """
        :param task: name of the task
        :param data_path: path to datafile
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param index_col: column to use as index
        :param target_col: name of the targeted column
        :param seq_length: window length to use
        :param batch_size:
        :param prediction_window: window length to predict
        """
        self.task = task.value

        data_path = pkg_resources.resource_filename("tsa", data_path)
        self.data = pd.read_csv(data_path, index_col=index_col)
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(self.data.columns) - set(categorical_cols) - set(target_col))
        self.target_col = target_col

        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size

        transformations = [("scaler", StandardScaler(), self.numerical_cols)]
        if len(self.categorical_cols) > 0:
            transformations.append(("encoder", OneHotEncoder(), self.categorical_cols))
        self.preprocessor = ColumnTransformer(transformations, remainder="passthrough")

        if self.task == "prediction":
            self.y_scaler = StandardScaler()

    def preprocess_data(self):
        """Preprocessing function"""
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        if self.task == "prediction":
            y_train = self.y_scaler.fit_transform(y_train)
            y_test = self.y_scaler.transform(y_test)
            return X_train, X_test, y_train, y_test
        return X_train, X_test, None, None

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.seq_length - self.prediction_window):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

            if self.task == "prediction":
                # lagged output used for prediction
                y_hist.append(torch.FloatTensor(y[i - 1:i + self.seq_length - 1]).unsqueeze(0))
                # shifted target
                target.append(torch.FloatTensor(y[i + self.seq_length:i + self.seq_length + self.prediction_window]))
            else:
                y_hist.append(torch.FloatTensor(X[i - 1: i + self.seq_length - 1, :]).unsqueeze(0))
                target.append(
                    torch.FloatTensor(X[i + self.seq_length:i + self.seq_length + self.prediction_window, :]))

        features_var = torch.cat(features)
        y_hist_var = torch.cat(y_hist)
        target_var = torch.cat(target)

        return TensorDataset(features_var, y_hist_var, target_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset

        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """
        Inverts the scale of the predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if self.task == "prediction":
            unscaled = self.y_scaler.inverse_transform(predictions)
        else:
            unscaled = self.preprocessor.named_transformers_["scaler"].inverse_transform(predictions)
        return torch.Tensor(unscaled)


class LOBSTERTimeSeriesDataset(object):
    """Data wrapper around :mod:`lobster_toolkit` processed outputs.

    This class bridges the project autoencoder expectations – returning
    ``(features, y_hist, target)`` tensors – with the pre-processing pipeline
    extracted to ``export/lobster_toolkit``.  It supports both reconstruction
    and prediction tasks by reusing the sliding window logic implemented for
    the baseline :class:`TimeSeriesDataset`.
    """

    _VALID_TARGET_SOURCES = {"features", "labels", "labels_long", "labels_short"}

    def __init__(
        self,
        task: Tasks,
        data_path: str,
        seq_length: int,
        batch_size: int,
        prediction_window: int = 1,
        target_source: str = "features",
        message_columns: Optional[Iterable[str]] = None,
        orderbook_columns: Optional[Iterable[str]] = None,
        build_if_missing: bool = False,
        builder_kwargs: Optional[Dict[str, Any]] = None,
        label_type: str = "long",
        shuffle_train: bool = False,
        prefer_val_split: bool = True,
    ) -> None:
        self.task = task.value
        if target_source not in self._VALID_TARGET_SOURCES:
            raise ValueError(
                f"target_source 必须是 {sorted(self._VALID_TARGET_SOURCES)} 之一，当前为 {target_source}"
            )

        if target_source.startswith("labels") and label_type not in {"long", "short"}:
            raise ValueError("label_type 必须是 'long' 或 'short'")

        if _to_absolute_path is not None:
            self.root_path = os.path.abspath(_to_absolute_path(data_path))
        else:
            self.root_path = os.path.abspath(data_path)
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.target_source = target_source
        self.message_columns = None if message_columns is None else list(message_columns)
        self.orderbook_columns = None if orderbook_columns is None else list(orderbook_columns)
        self.label_type = label_type
        self.shuffle_train = shuffle_train
        self.prefer_val_split = prefer_val_split

        if build_if_missing and not self._has_processed_files():
            builder = LOBSTERDataBuilder(**(builder_kwargs or {}))
            builder.build_dataset(self.root_path)

        self.datasets = LOBSTERDataBuilder.load_processed_data(self.root_path)
        if not self.datasets:
            raise FileNotFoundError(
                f"未在 {self.root_path} 找到处理后的 LOBSTER 数据。"
                "请先运行 LOBSTERDataBuilder.build_dataset 或设置 build_if_missing=True。"
            )

    def _has_processed_files(self) -> bool:
        return os.path.exists(os.path.join(self.root_path, "train.pkl"))

    def preprocess_data(self) -> Tuple[TensorDataset, TensorDataset, int]:
        train_items = self.datasets.get("train", [])
        if not train_items:
            raise ValueError("train split 为空，无法构建数据集")

        train_dataset = self._frame_split(train_items)

        eval_items: Optional[List[Dict[str, Any]]] = None
        if self.prefer_val_split:
            eval_items = self.datasets.get("val")
        if not eval_items:
            eval_items = self.datasets.get("test")
        if not eval_items:
            eval_items = train_items

        test_dataset = self._frame_split(eval_items)

        if train_dataset.tensors:
            nb_features = train_dataset.tensors[0].shape[-1]
            self.target_size = train_dataset.tensors[2].shape[-1]
        else:
            nb_features = 0
            self.target_size = 0
        return train_dataset, test_dataset, nb_features

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, int]:
        train_dataset, test_dataset, nb_features = self.preprocess_data()

        train_iter = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            drop_last=True,
        )
        test_iter = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return train_iter, test_iter, nb_features

    def _select_columns(self, df: pd.DataFrame, wanted: Optional[List[str]]) -> pd.DataFrame:
        if wanted is None:
            return df
        missing = set(wanted) - set(df.columns)
        if missing:
            raise KeyError(f"以下列在数据集中不存在: {sorted(missing)}")
        return df[wanted]

    def _combine_features(self, item: Dict[str, Any]) -> np.ndarray:
        message_df = self._select_columns(item["message"], self.message_columns)
        orderbook_df = self._select_columns(item["orderbook"], self.orderbook_columns)
        combined = pd.concat([message_df, orderbook_df], axis=1)
        return combined.to_numpy(dtype=np.float32)

    def _get_target_series(self, item: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        if self.target_source == "features":
            return features

        if self.target_source == "labels":
            key = "labels_long" if self.label_type == "long" else "labels_short"
        elif self.target_source.endswith("long"):
            key = "labels_long"
        else:
            key = "labels_short"

        labels = item[key]
        target = np.asarray(labels, dtype=np.float32)
        return target[:, None] if target.ndim == 1 else target

    def _frame_split(self, items: List[Dict[str, Any]]) -> TensorDataset:
        features_batches: List[torch.Tensor] = []
        y_hist_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []

        for item in items:
            features = self._combine_features(item)
            targets = self._get_target_series(item, features)

            seq_source = targets if self.task == Tasks.prediction.value else features

            length = min(len(features), len(targets))
            max_index = length - self.seq_length - self.prediction_window
            if max_index <= 0:
                continue

            features = features[:length]
            targets = targets[:length]
            seq_source = seq_source[:length]

            for start in range(1, max_index + 1):
                end = start + self.seq_length

                feature_window = torch.from_numpy(features[start:end]).float()
                features_batches.append(feature_window.unsqueeze(0))

                y_hist_window = torch.from_numpy(seq_source[start - 1:end - 1]).float()
                y_hist_batches.append(y_hist_window.unsqueeze(0))

                target_slice = torch.from_numpy(targets[end : end + self.prediction_window]).float()
                target_batches.append(target_slice)

        if not features_batches:
            raise ValueError("给定的数据不足以构建任何序列，请检查 seq_length 或数据量")

        features_var = torch.cat(features_batches)
        y_hist_var = torch.cat(y_hist_batches)
        target_tensor = torch.stack(target_batches, dim=0)
        target_var = target_tensor.view(target_tensor.size(0), -1)
        if not hasattr(self, "_target_size"):
            self._target_size = target_var.shape[-1]

        return TensorDataset(features_var, y_hist_var, target_var)

    @staticmethod
    def invert_scale(predictions: torch.Tensor) -> torch.Tensor:
        """Identity helper to mirror :class:`TimeSeriesDataset` API."""

        return predictions.detach() if isinstance(predictions, torch.Tensor) else torch.as_tensor(predictions)
