<h1 align="center">LSTM-autoencoder with attentions for multivariate time series</h1>

<p align="center">
    <img src="https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2FJulesBelveze%2Ftime-series-autoencoder" alt="Hits">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg">
</p>

This repository contains an autoencoder for multivariate time series forecasting.
It features two attention mechanisms described
in *[A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)*
and was inspired by [Seanny123's repository](https://github.com/Seanny123/da-rnn).

![Autoencoder architecture](autoenc_architecture.png)

## Download and dependencies

To clone the repository please run:

```
git clone https://github.com/JulesBelveze/time-series-autoencoder.git
```

<details>

<summary>Use uv</summary>

Then install `uv` 
```shell
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # linux/mac
# or
brew install uv  # mac with homebrew
```

# setup environment and install dependencies
```bash
cd time-series-autoencoder
uv venv
uv pip sync pyproject.toml
```

</details>

<details>
<summary>Install directly from requirements.txt</summary>

```shell
pip install -r requirements.txt
```

</details>

## Usage

The project uses [Hydra](https://hydra.cc/docs/intro/) as a configuration parser. You can simply change the parameters
directly within your `.yaml` file or you can override/set parameter using flags (for a complete guide please refer to
the docs).

```
python3 main.py -cn=[PATH_TO_FOLDER_CONFIG] -cp=[CONFIG_NAME]
```

Optional arguments:

```  
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --output-size OUTPUT_SIZE
                        size of the ouput: default value to 1 for forecasting
  --label-col LABEL_COL
                        name of the target column
  --input-att INPUT_ATT
                        whether or not activate the input attention mechanism
  --temporal-att TEMPORAL_ATT
                        whether or not activate the temporal attention
                        mechanism
  --seq-len SEQ_LEN     window length to use for forecasting
  --hidden-size-encoder HIDDEN_SIZE_ENCODER
                        size of the encoder's hidden states
  --hidden-size-decoder HIDDEN_SIZE_DECODER
                        size of the decoder's hidden states
  --reg-factor1 REG_FACTOR1
                        contribution factor of the L1 regularization if using
                        a sparse autoencoder
  --reg-factor2 REG_FACTOR2
                        contribution factor of the L2 regularization if using
                        a sparse autoencoder
  --reg1 REG1           activate/deactivate L1 regularization
  --reg2 REG2           activate/deactivate L2 regularization
  --denoising DENOISING
                        whether or not to use a denoising autoencoder
  --do-train DO_TRAIN   whether or not to train the model
  --do-eval DO_EVAL     whether or not evaluating the mode
  --data-path DATA_PATH
                        path to data file
  --output-dir OUTPUT_DIR
                        name of folder to output files
  --ckpt CKPT           checkpoint path for evaluation 
  ```

## Features

* handles multivariate time series
* attention mechanisms
* denoising autoencoder
* sparse autoencoder

## 项目概览（中文）

* 代码主体位于 `tsa/` 包中，其中 `model.py` 定义了编码器、解码器以及整体的自编码器模型，`train.py`、`eval.py` 与 `dataset.py` 分别负责训练流程、评估逻辑和数据读取。
* 示例脚本放在 `examples/` 目录下，`forecasting/` 与 `reconstruction/` 子目录提供了基于 Hydra 的配置文件及运行入口，可直接复现实验。
* 根目录中的 `autoenc_architecture.png` 对模型结构进行了可视化说明，`pyproject.toml` 与 `requirements.txt` 列出了依赖。
* 若要接入 `export/` 目录中的数据预处理模块，请注意该目录并未包含在本仓库中，需要从原项目单独引入。

## 模型结构详解（中文）

* **编码器（`tsa/model.py::Encoder` 与 `AttnEncoder`）**：基础版本是单层 LSTM，会按时间步迭代更新隐藏状态。若在配置中启用 `input_att`，则切换为 `AttnEncoder`，它会在每个时间步根据当前隐藏状态和输入序列计算输入注意力权重，并按权重对各个特征逐元素缩放后直接送入 LSTM。`denoising` 选项打开时，编码器在训练阶段会按概率给输入添加高斯噪声，实现去噪自编码器的效果。
* **解码器（`tsa/model.py::Decoder` 与 `AttnDecoder`）**：默认解码器是单层 LSTM，通过历史目标序列重构下一个时间步的输出。若启用 `temporal_att`，则使用 `AttnDecoder`：它会对编码器输出的各时间步表示计算时间注意力，得到上下文向量后与历史目标拼接，再经过全连接层生成候选输入并更新解码 LSTM，最终输出预测值。
* **整体模型（`tsa/model.py::AutoEncForecast`）**：根据配置选择是否使用输入注意力和时间注意力，将编码器输出传递给相应的解码器。模型支持多变量时间序列，窗口长度由 `seq_len` 控制，隐藏维度则由 `hidden_size_encoder` 与 `hidden_size_decoder` 指定。
* **可选正则与损失设置**：配置文件中可以启用 `reg1` / `reg2` 进行 L1/L2 正则化，并通过 `reg_factor1`、`reg_factor2` 控制权重；`denoising`、`input_att`、`temporal_att` 等布尔开关可在命令行中覆盖，方便做消融实验。

## Examples

You can find under the `examples` scripts to train the model in both cases:

* reconstruction: the dataset can be found [here](https://gist.github.com/JulesBelveze/99ecdbea62f81ce647b131e7badbb24a)
* forecasting: the dataset can be found [here](https://gist.github.com/JulesBelveze/e9997b9b0b68101029b461baf698bd72)
  
  
