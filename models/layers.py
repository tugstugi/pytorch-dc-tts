__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['E', 'D', 'C', 'HighwayBlock', 'GatedConvBlock', 'ResidualBlock']

import torch.nn as nn
import torch.nn.functional as F

from hparams import HParams as hp


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """Layer Norm."""
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # PyTorch LayerNorm seems to be expect (B, T, C)
        y = super(LayerNorm, self).forward(x)
        y = y.permute(0, 2, 1)  # reverse
        return y


class D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, weight_init='none', normalization='weight', nonlinearity='linear'):
        """1D Deconvolution."""
        super(D, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                         stride=2,  # paper: stride of deconvolution is always 2
                                         dilation=dilation)

        if normalization == 'weight':
            self.deconv = nn.utils.weight_norm(self.deconv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.deconv.weight, nn.init.calculate_gain(nonlinearity))

    def forward(self, x, output_size=None):
        y = self.deconv(x, output_size=output_size)
        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, causal=False, weight_init='none', normalization='weight', nonlinearity='linear'):
        """1D convolution.
        The argument 'causal' indicates whether the causal convolution should be used or not.
        """
        super(C, self).__init__()
        self.causal = causal
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1,  # paper: 'The stride of convolution is always 1.'
                              padding=self.padding, dilation=dilation)

        if normalization == 'weight':
            self.conv = nn.utils.weight_norm(self.conv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain(nonlinearity))

    def forward(self, x):
        y = self.conv(x)
        padding = self.padding
        if self.causal and padding > 0:
            y = y[:, :, :-padding]

        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


class E(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(E, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


class HighwayBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight'):
        """Highway Network like layer: https://arxiv.org/abs/1505.00387
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(HighwayBlock, self).__init__()
        self.d = d
        self.C = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta, causal=causal, weight_init=weight_init, normalization=normalization)

    def forward(self, x):
        L = self.C(x)
        H1 = L[:, :self.d, :]
        H2 = L[:, self.d:, :]
        sigH1 = F.sigmoid(H1)
        return sigH1 * H2 + (1 - sigH1) * x


class GatedConvBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight'):
        """Gated convolutional layer: https://arxiv.org/abs/1612.08083
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(GatedConvBlock, self).__init__()
        self.C = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta, causal=causal,
                   weight_init=weight_init, normalization=normalization)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        L = self.C(x)
        return self.glu(L) + x


class ResidualBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight',
                 widening_factor=2):
        """Residual block: https://arxiv.org/abs/1512.03385
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(ResidualBlock, self).__init__()
        self.C1 = C(in_channels=d, out_channels=widening_factor * d, kernel_size=k, dilation=delta, causal=causal,
                    weight_init=weight_init, normalization=normalization, nonlinearity='relu')
        self.C2 = C(in_channels=widening_factor * d, out_channels=d, kernel_size=k, dilation=delta, causal=causal,
                    weight_init=weight_init, normalization=normalization, nonlinearity='relu')

    def forward(self, x):
        return self.C2(self.C1(x)) + x
