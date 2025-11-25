import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden state tensor directly on the target device.

    Args:
        x: (torch.Tensor): input tensor (used for batch size and device)
        hidden_size: (int): hidden state size
        num_dir: (int): number of directions in LSTM
        xavier: (bool): whether to use xavier initialization
    """
    h = torch.zeros(num_dir, x.size(0), hidden_size, device=x.device)
    if xavier:
        nn.init.xavier_normal_(h)
    return h


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config['hidden_size_encoder'])

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        # 预分配输出 tensor 在正确的 device 上
        input_encoded = torch.zeros(
            input_data.size(0), self.seq_len, self.hidden_size, 
            device=input_data.device
        )

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t
        return _, input_encoded


class AttnEncoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.add_noise = config['denoising']
        self.directions = config['directions']
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _get_noise(input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise directly on the input device.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape, device=input_data.device)
        mask = torch.rand(input_data.shape, device=input_data.device) < p
        return normal * mask.float()

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation with optimized device handling.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        batch_size = input_data.size(0)
        dev = input_data.device
        
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        # 预分配输出 tensor 在正确的 device 上
        attentions = torch.zeros(batch_size, self.seq_len, self.input_size, device=dev)
        input_encoded = torch.zeros(batch_size, self.seq_len, self.hidden_size, device=dev)

        if self.add_noise and self.training:
            input_data = input_data + self._get_noise(input_data)

        # 预计算 input_data 的转置（避免循环内重复计算）
        input_permuted = input_data.permute(0, 2, 1)  # (batch, input_size, seq_len)

        for t in range(self.seq_len):
            # 拼接 attention 输入（所有数据已在同一 device）
            x = torch.cat((
                h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                input_permuted
            ), dim=2)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size))  # (bs, input_size)

            weighted_input = a_t * input_data[:, t, :]  # (bs, input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = config['seq_len']
        self.hidden_size = config['hidden_size_decoder']
        self.lstm = nn.LSTM(1, config['hidden_size_decoder'], bidirectional=False)
        self.fc = nn.Linear(config['hidden_size_decoder'], config['output_size'])

    def forward(self, _, y_hist: torch.Tensor):
        """
        Forward pass

        Args:
            _: unused encoder output
            y_hist: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(y_hist, self.hidden_size),
                    init_hidden(y_hist, self.hidden_size))

        for t in range(self.seq_len):
            inp = y_hist[:, t].unsqueeze(0).unsqueeze(2)
            lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        return self.fc(lstm_out.squeeze(0))


class AttnDecoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = config['seq_len']
        self.encoder_hidden_size = config['hidden_size_encoder']
        self.decoder_hidden_size = config['hidden_size_decoder']
        self.out_feats = config['output_size']

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation with optimized device handling.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        batch_size = input_encoded.size(0)
        dev = input_encoded.device
        
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size),
            init_hidden(input_encoded, self.decoder_hidden_size)
        )
        context = torch.zeros(batch_size, self.encoder_hidden_size, device=dev)

        for t in range(self.seq_len):
            # 所有数据已在同一 device，无需循环内调用 .to(device)
            x = torch.cat((
                h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                input_encoded
            ), dim=2)

            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1
            )

            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context), dim=1))  # predicting value at t=self.seq_length+1


class AutoEncForecast(nn.Module):
    def __init__(self, config, input_size):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(config, input_size).to(device) if config['input_att'] else \
            Encoder(config, input_size).to(device)
        self.decoder = AttnDecoder(config).to(device) if config['temporal_att'] else Decoder(config).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        if return_attention:
            return outputs, attentions
        return outputs
