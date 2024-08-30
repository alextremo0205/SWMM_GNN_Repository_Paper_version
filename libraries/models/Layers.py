"""
@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import torch
import torch.nn as nn


class FullyConnectedNN(nn.Module):
    """
    A fully connected neural network implemented as a PyTorch module.

    Args:
        in_dims (int): The number of input dimensions for the neural network.
        out_dims (int): The number of output dimensions for the neural network.
        hidden_dim (int): The number of hidden units in each hidden layer.
        n_hidden_layers (int): The number of hidden layers in the neural network.
        with_bias (bool, optional): Whether to include bias terms in linear layers. Defaults to True.
        non_linearity (str, optional): The non-linearity to apply after each linear layer.
                                      Supported values are "Identity", "ReLU", "Sigmoid", and more.
                                      Defaults to "Identity".
        final_bias (bool, optional): Whether to include a bias term in the final linear layer. Defaults to True.
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        hidden_dim,
        n_hidden_layers,
        with_bias=True,
        non_linearity="Identity",
        final_bias=True,
    ):

        super(FullyConnectedNN, self).__init__()
        self.with_bias = with_bias
        self.non_linearity = non_linearity
        self.linear_stack = nn.Sequential()

        self._append_block(in_dims, hidden_dim, bias=self.with_bias)

        for _ in range(n_hidden_layers):
            self._append_block(hidden_dim, hidden_dim, bias=self.with_bias)

        self._append_block(hidden_dim, out_dims, bias=final_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_stack(x)
        return x

    def _append_block(self, in_dim, out_dim, bias=True):
        self.linear_stack.append(nn.Linear(in_dim, out_dim, bias=bias))
        self.linear_stack.append(getattr(nn, self.non_linearity)())
