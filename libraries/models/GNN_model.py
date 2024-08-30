"""
# This module contains the models with a Encoder Processor Decoder architecture. These models also consider the elevation as input feature in the nodes.
# The convention NN in the name indicates a fully connected neural network in the encoder and the decoder.

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import copy
import torch
import torch.nn as nn

from torch_geometric.utils import scatter
from torch_geometric.nn import GINEConv

from libraries.models.Layers import FullyConnectedNN


class EPDBaseModel_with_elevation(nn.Module):
    """
    This class extends the `nn.Module` class and defines a forward method that takes a window `win` object as input
    and returns a tensor with the predicted hydraulic head at each time step.

    The model uses a set of equations that describe the behavior of the hydraulic system and PyTorch to perform
    the computations.

    Methods:
        forward(win): Performs a simulation of a hydraulic system using the input `win` and returns the predicted
            values of the hydraulic head at each time step.
        count_parameters(): Returns the total number of trainable parameters in the model.
    """

    def forward(self, win):
        win = copy.deepcopy(win)
        is_batch = win.batch is not None

        h0 = win.norm_h_x
        ground_level = win.norm_ground_level

        length_simulation = win.steps_ahead[0].item() if is_batch else win.steps_ahead
        self._assert_valid_length(length_simulation)

        pred_h_acum = torch.zeros(win.num_nodes, length_simulation)

        for step in range(0, length_simulation, self.prediction_steps):
            transformed_x = self._transform_x_with_layers(win, h0, step)
            pred_head = self._add_skip_connection(win, h0, transformed_x)

            pred_head_clipped = torch.min(pred_head, ground_level)
            pred_head_lower_clipped = torch.max(pred_head_clipped, win.norm_elevation)

            pred_h_acum[:, step : step + self.prediction_steps] = (
                pred_head_lower_clipped
            )

            h0 = self._get_new_h0(h0, pred_head_clipped)

        return pred_h_acum

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _transform_x_with_layers(self, win, h0, step):
        if self.aggregation_type == "Combined":
            transformed_x = self._use_layers_in_forward_pass_combined(win, h0, step)
        elif self.aggregation_type == "Separated":
            transformed_x = self._use_layers_in_forward_pass_separated(win, h0, step)
        else:
            raise Exception("Unknown aggregation")
        return transformed_x

    def _add_skip_connection(self, win, h0, transformed_x):
        prev_y = h0[:, -1].reshape(-1, 1) - win.norm_elevation
        pred_y_skipped = (
            self.skip_alpha * transformed_x + (1.0 - self.skip_alpha) * prev_y
        )
        pred_head = pred_y_skipped + win.norm_elevation
        return pred_head

    def _use_layers_in_forward_pass_separated(self, win, h0, step):
        edge_features = self._get_edge_features(win)
        node_features = self._get_one_step_features_node(win, h0, step)

        coded_x = self.layers_dict["nodeEncoder"](node_features)
        coded_e_i = self.layers_dict["edgeEncoder"](edge_features)

        processed_x = self.layers_dict["processor"](coded_x, win.edge_index, coded_e_i)

        decoded_x = self.layers_dict["nodeDecoder"](processed_x)
        return decoded_x

    def _use_layers_in_forward_pass_combined(self, win, h0, step):
        edge_features = self._get_edge_features(win)
        node_features = self._get_one_step_features_node(win, h0, step)

        source, target = win.edge_index

        node_features_in_source = node_features[source]
        node_features_in_target = node_features[target]

        mixed_features = torch.cat(
            [edge_features, node_features_in_source, node_features_in_target], axis=1
        )

        coded_e_i = self.layers_dict["edgeEncoderMix"](mixed_features)
        coded_x = scatter(coded_e_i, source, dim=0, reduce="sum")

        processed_x = self.layers_dict["processor"](coded_x, win.edge_index)

        decoded_x = self.layers_dict["nodeDecoder"](processed_x)
        return decoded_x

    def _get_one_step_features_node(self, win, h0, step):
        runoff_step = win.norm_runoff[
            :, step : step + self.steps_behind + self.prediction_steps
        ]

        one_step_x = torch.cat((h0, runoff_step, win.norm_elevation), dim=1)
        return one_step_x

    def _get_edge_features(self, win):
        edge_attributes = [win["norm_" + atr] for atr in self.edge_input_list]
        return torch.concat(edge_attributes, axis=1)

    def _assert_valid_length(self, length_simulation):
        assert (
            length_simulation >= self.prediction_steps
        ), "The prediction is longer than the desired simulation length."
        assert (
            length_simulation % self.prediction_steps == 0
        ), "The prediction should be a multiple of the simulation length."

    def _get_new_h0(self, old_h0, new_h):
        original_size = old_h0.shape[1]
        concatenated = torch.cat((old_h0, new_h), dim=1)
        new_h0 = concatenated[:, -original_size:]
        return new_h0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class NN_GINEConv_NN(EPDBaseModel_with_elevation):
    """
    This class extends the `EPDBaseModel_with_elevation` class and defines the layers of the model.
    This includes the node encoder, edge encoder, processor, and node decoder.
    """

    def __init__(
        self, steps_behind, hidden_dim, skip_alpha, prediction_steps=1, **kwargs
    ):
        super(NN_GINEConv_NN, self).__init__()
        self.aggregation_type = "Separated"

        self.steps_behind = steps_behind
        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps

        self.skip_alpha = nn.Parameter(torch.tensor(skip_alpha))
        self.length_window = (
            2 * self.steps_behind
        ) + self.prediction_steps  # Steps behind (depth), steps behind (runoff), prediction_steps (runoff)
        self.non_linearity = kwargs["non_linearity"]
        self.n_hidden_layers = kwargs["n_hidden_layers"]
        self.eps_gnn = kwargs["eps_gnn"]
        self.edge_input_list = kwargs["edge_input_list"]
        self.number_edge_inputs = len(self.edge_input_list)
        self.create_layers_dict()

    def create_layers_dict(self):
        self._nodeEncoder = FullyConnectedNN(
            self.length_window + 1,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )
        self._edgeEncoder = FullyConnectedNN(
            self.number_edge_inputs,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )

        _mlp_for_gineconv = FullyConnectedNN(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )

        self._processor = GINEConv(
            _mlp_for_gineconv, eps=self.eps_gnn, train_eps=True
        )  # .jittable()

        self._nodeDecoder = FullyConnectedNN(
            self.hidden_dim,
            self.prediction_steps,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
            final_bias=False,
        )

        self.layers_dict = nn.ModuleDict(
            {
                "nodeEncoder": self._nodeEncoder,
                "edgeEncoder": self._edgeEncoder,
                "processor": self._processor,
                "nodeDecoder": self._nodeDecoder,
            }
        )
