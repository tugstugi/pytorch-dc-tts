"""
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
https://arxiv.org/abs/1710.08969

SSRN Network.
"""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['SSRN']

import torch.nn as nn
import torch.nn.functional as F

from hparams import HParams as hp
from .layers import D, C, HighwayBlock, GatedConvBlock, ResidualBlock


def Conv(in_channels, out_channels, kernel_size, dilation, nonlinearity='linear'):
    return C(in_channels, out_channels, kernel_size, dilation, causal=False,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def DeConv(in_channels, out_channels, kernel_size, dilation, nonlinearity='linear'):
    return D(in_channels, out_channels, kernel_size, dilation,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def BasicBlock(d, k, delta):
    if hp.ssrn_basic_block == 'gated_conv':
        return GatedConvBlock(d, k, delta, causal=False,
                              weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    elif hp.ssrn_basic_block == 'highway':
        return HighwayBlock(d, k, delta, causal=False,
                            weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    else:
        return ResidualBlock(d, k, delta, causal=False,
                             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization,
                             widening_factor=1)


class SSRN(nn.Module):
    def __init__(self, c=hp.c, f=hp.n_mels, f_prime=(1 + hp.n_fft // 2)):
        """Spectrogram super-resolution network.
        Args:
            c: SSRN dim
            f: Number of mel bins
            f_prime: full spectrogram dim
        Input:
            Y: (B, f, T) predicted melspectrograms
        Outputs:
            Z_logit: logit of Z
            Z: (B, f_prime, 4*T) full spectrograms
        """
        super(SSRN, self).__init__()
        self.layers = nn.Sequential(
            Conv(f, c, 1, 1),

            BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),
            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            Conv(c, 2 * c, 1, 1),

            BasicBlock(2 * c, 3, 1), BasicBlock(2 * c, 3, 1),

            Conv(2 * c, f_prime, 1, 1),

            # Conv(f_prime, f_prime, 1, 1, nonlinearity='relu'),
            # Conv(f_prime, f_prime, 1, 1, nonlinearity='relu'),
            BasicBlock(f_prime, 1, 1),

            Conv(f_prime, f_prime, 1, 1)
        )

    def forward(self, x):
        Z_logit = self.layers(x)
        Z = F.sigmoid(Z_logit)
        return Z_logit, Z