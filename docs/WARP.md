# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Commands

### Environment Setup
```powershell
# Clone the repository
git clone https://github.com/JulesBelveze/time-series-autoencoder.git
cd time-series-autoencoder

# Using uv (recommended)
uv venv
uv pip sync pyproject.toml

# Using pip
pip install -r requirements.txt
```

### Running Training

```powershell
# Forecasting task (predict future values)
python examples/forecasting/run_forecasting.py

# Reconstruction task (reconstruct input sequences)
python examples/reconstruction/run_reconstruction.py

# Override config parameters via command line
python examples/forecasting/run_forecasting.py training.lr=1e-4 training.num_epochs=50

# Use custom config file
python examples/forecasting/run_forecasting.py --config-name=config_lobster
```

### Evaluation Only

```powershell
# Evaluate from checkpoint
python examples/forecasting/run_forecasting.py general.do_train=False general.ckpt=output/checkpoint-5000.ckpt
```

### Monitoring Training

```powershell
# Start TensorBoard to monitor training progress
tensorboard --logdir=output
```

## High-Level Architecture

### Model Overview

The project implements a dual-stage attention-based LSTM autoencoder for multivariate time series, featuring:
- **Encoder**: Processes input sequences, optionally with input attention mechanism
- **Decoder**: Generates predictions/reconstructions, optionally with temporal attention mechanism
- Two distinct tasks: **prediction** (forecasting) and **reconstruction**

### Key Architecture Decisions

**Encoder-Decoder Flow**:
1. Input: `(batch_size, seq_len, input_size)` feature tensor and `(batch_size, seq_len)` historical target
2. Encoder produces hidden state sequence: `(batch_size, seq_len, hidden_size_encoder)`
3. Decoder consumes encoder output and generates: `(batch_size, output_size)`

**Attention Mechanisms**:
- **Input Attention** (`AttnEncoder`): At each timestep, computes attention weights over input features based on current LSTM hidden state. The weighted input is then fed into the LSTM.
- **Temporal Attention** (`AttnDecoder`): At each decoding timestep, computes attention weights over all encoder hidden states to create a context vector that guides prediction.

**Model Selection Logic** (in `AutoEncForecast.__init__`):
- `config['input_att'] = True` → Use `AttnEncoder`, else use basic `Encoder`
- `config['temporal_att'] = True` → Use `AttnDecoder`, else use basic `Decoder`

### Core Modules

**`tsa/model.py`**:
- `Encoder` / `AttnEncoder`: Basic vs. attention-based encoding
- `Decoder` / `AttnDecoder`: Basic vs. attention-based decoding
- `AutoEncForecast`: Main model that combines encoder + decoder based on config flags

**`tsa/dataset.py`**:
- `TimeSeriesDataset`: Handles CSV data loading, preprocessing (StandardScaler, OneHotEncoder), and sliding window framing
- `LOBSTERTimeSeriesDataset`: Specialized dataset for LOBSTER financial data (requires `export/lobster_toolkit` module)
- `Tasks` enum: `prediction` vs `reconstruction`
- **Key method**: `frame_series()` creates sliding windows with `seq_length` lookback and returns `(features, y_hist, target)` tensors

**`tsa/train.py`**:
- Main training loop with TensorBoard logging
- Supports L1/L2 regularization (sparse autoencoder)
- Gradient accumulation and clipping
- StepLR scheduler with configurable step size
- Periodic evaluation and checkpoint saving

**`tsa/eval.py`**:
- Computes test loss, MSE, and residuals
- Generates prediction plots (`preds.png`)
- Saves predictions, targets, and attention weights as `.pt` files

### Configuration System

Uses **Hydra** for hierarchical configuration:
- Config files are in `examples/{forecasting,reconstruction}/config.yaml`
- Three main sections: `data`, `training`, `general`
- Key parameters:
  - `data.task.value`: `"prediction"` or `"reconstruction"`
  - `data.target_col`: Target column(s) for prediction task; empty list for reconstruction
  - `training.input_att`, `training.temporal_att`: Enable attention mechanisms
  - `training.denoising`: Add Gaussian noise to encoder input during training
  - `training.output_size`: For prediction, should match target dimension; for reconstruction, should match input feature count
  - `general.eval_during_training`: Whether to run evaluation during training

### Data Flow Differences

**Prediction Task**:
- `target_col` is specified in config
- `TimeSeriesDataset` uses `y_scaler` to normalize targets
- `y_hist` = lagged target values `y[i-1:i+seq_length-1]`
- `target` = future target value `y[i+seq_length:i+seq_length+prediction_window]`

**Reconstruction Task**:
- `target_col` is empty list
- No target normalization
- `y_hist` = lagged features `X[i-1:i+seq_length-1, :]`
- `target` = current window features `X[i+seq_length:i+seq_length+prediction_window, :]`

### Output Structure

After training, `output/` directory contains:
- `checkpoint-*.ckpt`: Model checkpoints (encoder, decoder, optimizer state)
- `preds.png`: Prediction vs ground truth plot
- `predictions.pt`, `targets.pt`, `attentions.pt`: Raw tensors
- `eval_results.txt`: MSE, residuals, and loss metrics
- TensorBoard event files for train/test metrics

### LOBSTER Data Processing Pipeline

This repository includes a complete **`export/lobster_toolkit`** module for processing high-frequency LOBSTER (Limit Order Book System) trading data. The pipeline is separate from the main autoencoder but fully integrated.

#### LOBSTER Toolkit Architecture

**Location**: `export/lobster_toolkit/`

**Core Modules** (no framework dependencies):
- **`builder.py`**: `LOBSTERDataBuilder` - Main orchestrator for the entire pipeline
- **`preprocessor.py`**: `MessagePreprocessor`, `OrderbookPreprocessor` - Load and preprocess raw CSV files
- **`labeler.py`**: `StopLossLabelStrategy` - Generate binary labels based on stop-loss/take-profit logic
- **`normalizer.py`**: `ZScoreNormalizer` - Z-score normalization with parameter persistence

**Adapters** (framework-specific):
- **`loaders/pytorch.py`**: PyTorch DataLoader integration
- **`adapters/aeon.py`**: Time series classification framework adapter

#### Data Processing Flow

1. **File Discovery & Pairing**: 
   - Scans for `*_message_*.csv` and `*_orderbook_*.csv` file pairs in `data/{data_dir_name}/` (default: `GC/`)
   - Each message file must have a matching orderbook file with identical date/identifier

2. **Dataset Splitting**:
   - Files are split by date into train/val/test according to `split_rates` (default: 0.8/0.1/0.1)
   - No shuffling - maintains temporal order

3. **Preprocessing** (per file pair):
   - **Message preprocessing**:
     - Load 6 columns: `time`, `type`, `order_id`, `size`, `price`, `direction`
     - Scale price by `/10000`
     - Compute time differences: `time[i] = time[i] - time[i-1]` (first row = 0)
     - Reverse direction sign: `direction *= -1`
     - Drop `type` and `order_id` columns
   - **Orderbook preprocessing**:
     - Load 40 columns: 10 levels of `ask_price`, `ask_size`, `bid_price`, `bid_size`
     - Scale all prices by `/10000`
   - **Relative price normalization**:
     - Subtract starting price from all message and orderbook prices
     - Makes prices relative to session start

4. **Label Generation** (`StopLossLabelStrategy`):
   - Generates **dual binary labels** for each timestamp: `labels_long` and `labels_short`
   - **Long strategy**: Label = 1 if take-profit (price rises by `take_profit_ratio * stop_loss_distance`) triggers before stop-loss (price falls by `stop_loss_distance`)
   - **Short strategy**: Opposite - Label = 1 if price falls to take-profit before rising to stop-loss
   - Parameters:
     - `stop_loss_ticks`: Stop-loss threshold in ticks (default: 6)
     - `take_profit_ratio`: Take-profit multiplier (default: 1.5)
     - `tick_size`: Minimum price movement (default: 0.1)
     - `max_steps`: Forward-looking window size (default: 64 timesteps)
   - Uses `best_bid` (orderbook's `bid_price_1`) and `best_ask` (orderbook's `ask_price_1`)

5. **Z-Score Normalization**:
   - Fitted on **train set only**, then applied to all splits
   - **Message columns normalized**: `time`, `size`
   - **Orderbook columns normalized**: All `*_size_*` columns (e.g., `ask_size_1`, `bid_size_2`)
   - Formula: `(x - mean) / std`
   - Saves parameters to `norm_params.csv`

6. **Output Files**:
   - `train.pkl`, `val.pkl`, `test.pkl`: Pickled lists of dicts, each containing:
     ```python
     {
         'date': str,                      # Date identifier from filename
         'message': pd.DataFrame,          # Preprocessed & normalized message data
         'orderbook': pd.DataFrame,        # Preprocessed & normalized orderbook data
         'labels_long': np.ndarray,        # Binary labels for long strategy
         'labels_short': np.ndarray        # Binary labels for short strategy
     }
     ```
   - `norm_params.csv`: Normalization means and stds for reproducibility

#### LOBSTERTimeSeriesDataset Integration

The `tsa/dataset.py::LOBSTERTimeSeriesDataset` class bridges the toolkit with the autoencoder:

**Key Parameters**:
- `data_path`: Root directory containing `train.pkl`, `val.pkl`, `test.pkl`
- `target_source`: Which data to use as target (`"features"`, `"labels"`, `"labels_long"`, `"labels_short"`)
- `label_type`: If `target_source="labels"`, choose `"long"` or `"short"`
- `message_columns`, `orderbook_columns`: Subset of columns to use (default: all)
- `build_if_missing`: Auto-run `LOBSTERDataBuilder` if pkl files don't exist
- `builder_kwargs`: Dict passed to `LOBSTERDataBuilder.__init__()` (e.g., `{"data_dir_name": "GC"}`)

**Data Flow**:
1. Loads processed pkl files via `LOBSTERDataBuilder.load_processed_data()`
2. For each data item, combines selected message/orderbook columns into `features` array
3. Selects `target` series based on `target_source`:
   - `"features"`: Target = features (reconstruction task)
   - `"labels*"`: Target = label array (prediction task)
4. Creates sliding windows using `_frame_split()` method:
   - `features_var`: `(n_windows, seq_length, n_features)`
   - `y_hist_var`: Historical sequence for decoder
     - Prediction task: `y_hist` = lagged targets `[start-1:end-1]`
     - Reconstruction task: `y_hist` = lagged features `[start-1:end-1]`
   - `target_var`: `(n_windows, target_dim)`
5. Sets `self.target_size` attribute, which `run_forecasting.py` reads to auto-configure `cfg.training.output_size`

**No Scaling in LOBSTERTimeSeriesDataset**: The toolkit already applies normalization, so `invert_scale()` is a no-op (returns input as-is).

#### Usage Example

```powershell
# Install toolkit (from export/ directory)
cd export
pip install -e .
cd ..

# Run with LOBSTER data
python examples/forecasting/run_forecasting.py --config-name=config_lobster
```

The `config_lobster.yaml` demonstrates LOBSTER-specific configuration:
- Uses `LOBSTERTimeSeriesDataset` instead of `TimeSeriesDataset`
- `target_source: features` for reconstruction task
- Longer `seq_length: 64` typical for high-frequency data
- `data_path: "../data/lobster"` expects `train.pkl`, `val.pkl`, `test.pkl` there
