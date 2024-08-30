"""
# This module contains the MLP benchmark model. This model is based on the architecture from Palmitessa et al. (2022).

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import torch
import torch.nn as nn
from libraries.models.Layers import FullyConnectedNN


class MLP_Benchmark_metamodel(nn.Module):
    """
    Tailor made model for "Tuindorp development" based on the architecture from Palmitessa et al. (2022).
    It considers nine steps behind for both runoff and hydraulic head, and next step ahead for runoff.  (9*2)+1
    This model considers recursive prediction. Clipping on the lower (inverts) and higher levels (ground level).
    The last non-linearity was changed to tanh since the paper from Palmitessa et al. did not specify the output non-linearity.

    """

    def __init__(self, steps_behind, prediction_steps=1, **kwargs):
        super(MLP_Benchmark_metamodel, self).__init__()
        NUM_NODES = 311
        self.steps_behind = steps_behind
        self.prediction_steps = prediction_steps
        self.layer = FullyConnectedNN(
            (steps_behind * 2 + 1) * NUM_NODES,
            NUM_NODES,
            hidden_dim=kwargs["hidden_dim"],
            n_hidden_layers=kwargs["n_hidden_layers"],
            non_linearity=kwargs["non_linearity"],
        )
        self.res_layer = FullyConnectedNN(
            NUM_NODES,
            NUM_NODES,
            hidden_dim=kwargs["hidden_dim"],
            n_hidden_layers=1,
            non_linearity=kwargs["non_linearity"],
        )

    def forward(self, win_batch):
        is_batch = win_batch.batch is not None
        length_simulation = (
            win_batch.steps_ahead[0].item() if is_batch else win_batch.steps_ahead
        )

        batch_size = len(win_batch.ptr) - 1 if is_batch else 1
        self._assert_valid_length(length_simulation)

        elevation = win_batch.norm_elevation.reshape(batch_size, -1)
        ground_level = win_batch.norm_ground_level.reshape(batch_size, -1)

        pred_h_acum = torch.zeros(win_batch.num_nodes, length_simulation)

        head_step = win_batch.norm_h_x.reshape(batch_size, -1)
        head_time_t = win_batch.norm_h_x[:, -1].reshape(batch_size, -1)

        for step in range(0, length_simulation, self.prediction_steps):
            runoff_step = self._get_runoff_extended_window_time(
                win_batch, step
            ).reshape(batch_size, -1)
            heads_and_runoff = torch.cat([head_step, runoff_step], axis=1)
            head_time_t = self.res_layer(head_time_t) + (
                self.layer(heads_and_runoff)
            )  # h + Δh nn.Tanh()

            head_time_t = torch.max(head_time_t, elevation)
            head_time_t = torch.min(head_time_t, ground_level)

            pred_h_acum[:, step : step + self.prediction_steps] = head_time_t.reshape(
                -1, 1
            )
            head_step = self._get_new_h0(head_step, head_time_t)

        return pred_h_acum

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_runoff_extended_window_time(self, win, step):
        start = step
        end = start + self.steps_behind + self.prediction_steps
        runoff_step = win.norm_runoff[:, start:end]
        # runoff_step = runoff_step[:,self.steps_behind].reshape(1,-1)
        return runoff_step

    def _get_new_h0(self, old_h0, y):
        original_size = old_h0.shape[1]
        concatenated = torch.cat((old_h0, y), dim=1)
        new_h0 = concatenated[:, -original_size:]
        return new_h0

    def _assert_valid_length(self, length_simulation):
        assert (
            length_simulation >= self.prediction_steps
        ), "The prediction is longer than the desired simulation length."
        assert (
            length_simulation % self.prediction_steps == 0
        ), "The prediction should be a multiple of the simulation length."
