
import numpy as np
import torch
import torch.nn as nn
import seqmodels as sm


class ResidualBind(nn.Module):

    def __init__(
        self,
        input_len,
        output_dim,
        input_channels=4,
        conv_channels=96,
        conv_kernel=11,
        conv_stride=1,
        conv_dilation=1,
        conv_padding="valid",
        conv_bias=False,
        conv_norm_type="batchnorm",
        conv_activation="relu",
        conv_dropout_rate=0.1,
        conv_order="conv-norm-act-dropout",
        num_residual_blocks=3,
        residual_channels=96,
        residual_kernel=3,
        residual_stride=1,
        residual_dilation_base=2,
        residual_biases=False,
        residual_activation="relu",
        residual_norm_type="batchnorm",
        residual_dropout_rates=0.1,
        residual_order="conv-norm-act-dropout",
        avg_pool_kernel=10,
        avg_pool_dropout_rate=0.2,
        dense_hidden_dims=[256],
        dense_biases=False,
        dense_activation="relu",
        dense_norm_type="batchnorm",
        dense_dropout_rates=0.5,
        dense_order="linear-norm-act-dropout",
    ):
        super(ResidualBind, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim

        self.conv1d_block = sm.Conv1DBlock(
            input_len=input_len,
            input_channels=input_channels,
            output_channels=conv_channels,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            conv_padding=conv_padding,
            conv_bias=conv_bias,
            activation=conv_activation,
            pool_type=None,
            dropout_rate=conv_dropout_rate,
            norm_type=conv_norm_type,
            order=conv_order,
        )
        self.res_tower = sm.layers.Residual(sm.Tower(
            input_size=(self.conv1d_block.output_size),
            block=sm.Conv1DBlock,
            repeats=num_residual_blocks,
            static_block_args={
                'input_len': self.conv1d_block.output_size[-1], 
                'output_channels': residual_channels,
                'conv_kernel': residual_kernel,
                'conv_stride': residual_stride,
                'conv_padding': 'same',
                'conv_bias': residual_biases,
                'activation': residual_activation,
                'pool_type': None,
                'norm_type': residual_norm_type,
                'dropout_rate': residual_dropout_rates,
                'order': residual_order,
            },
            dynamic_block_args={
                'input_channels': [conv_channels] + [residual_channels] * (num_residual_blocks - 1),
                'conv_dilation': [residual_dilation_base**i for i in range(num_residual_blocks)],
            }
        ))
        self.average_pool = nn.AvgPool1d(kernel_size=avg_pool_kernel, stride=1, padding=0)
        self.average_pool_dropout = nn.Dropout(p=avg_pool_dropout_rate)
        self.flatten = nn.Flatten()
        self.flatten_dim = self.res_tower.wrapped.output_size[-2] * (self.res_tower.wrapped.output_size[-1]-avg_pool_kernel+1)
        self.dense_tower = sm.Tower(
            input_size=self.flatten_dim,
            block=sm.DenseBlock,
            repeats=len(dense_hidden_dims)+1,
            static_block_args={
                'activation': dense_activation,
                'bias': dense_biases,
                'norm_type': dense_norm_type,
                'dropout_rate': dense_dropout_rates,
                'order': dense_order,
            },
            dynamic_block_args={
                'input_dim': [self.flatten_dim] + dense_hidden_dims,
                'output_dim': dense_hidden_dims+[1], 
                'dropout_rate': [0.5, None], 
                'order': ['linear-norm-act-dropout', 'linear']},
        )


    def forward(self, x):
        x = self.conv1d_block(x)
        x = self.res_tower(x)
        x = self.average_pool(x)
        x = self.average_pool_dropout(x)
        x = self.flatten(x)
        x = self.dense_tower(x)
        return x
