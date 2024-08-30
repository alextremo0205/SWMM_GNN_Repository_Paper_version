"""
This module provides normalization utilities for data preprocessing in machine learning experiments.

Classes:
    abstractNormalizer (ABC): An abstract base class for normalizers.
    Normalizer (abstractNormalizer): A normalizer that performs specific normalization on training windows.

Functions:
    NormalizerFactory(normalizer_name): Factory function to create normalizer instances.

Usage:
    normalizer = NormalizerFactory("Normalizer")(training_windows)
    normalized_window = normalizer.normalize_window(window)
    
@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod


def NormalizerFactory(normalizer_name):
    available_normalizer = {
        "Normalizer": Normalizer,
    }
    normalizer = available_normalizer[normalizer_name]
    return normalizer


class abstractNormalizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def normalize_window(self, window):
        pass


class Normalizer(abstractNormalizer):
    """
    This class normalizes the training windows for the machine learning model.
    It considers the following attributes for normalization:
    - h_x, h_y, elevation, ground_level, in_offset, out_offset
    - Optional: q_x, q_y in case of using flow rates
    """

    def __init__(self, training_windows, abs_flows=False):
        self.training_windows = training_windows
        self.abs_flows = abs_flows
        self.name_nodes = self.training_windows[0].name_nodes
        self.name_conduits = self.training_windows[0].name_conduits

        self.depths_normalizer = self._InternalNormalizer(
            self, ["h_x", "h_y", "elevation", "ground_level"]
        )
        self.flows_normalizer = self._InternalNormalizer(self, ["q_x", "q_y"])
        self.length_normalizer = self._InternalNormalizer(self, ["length"])
        self.geom_1_normalizer = self._InternalNormalizer(self, ["geom_1"])
        self.runoff_normalizer = self._InternalNormalizer(self, ["runoff"])
        self.slope_normalizer = self._InternalNormalizer(self, ["slope"])
        self.volume_normalizer = self._InternalNormalizer(
            self, ["aprox_conduit_volume"]
        )
        self.training_windows = None

    def normalize_window(self, window):
        head_attributes = [
            "h_x",
            "h_y",
            "elevation",
            "ground_level",
        ]  # 'in_offset', 'out_offset'
        depth_attributes = ["in_offset", "out_offset"]
        flow_attributes = ["q_x", "q_y"]

        for atr in head_attributes:
            window = self.depths_normalizer(window, atr)

        for atr in depth_attributes:
            window = self.depths_normalizer.scale_attribute(window, atr)

        for atr in flow_attributes:
            if self.abs_flows:
                window[atr] = abs(
                    window[atr]
                )  # ! This is done to avoid the direction of the flow given as sign
            window = self.flows_normalizer.scale_attribute(window, atr)

        window = self.length_normalizer(window, "length")
        window = self.geom_1_normalizer(window, "geom_1")
        window = self.runoff_normalizer(window, "runoff")
        window = self.slope_normalizer(window, "slope")
        window = self.volume_normalizer(window, "aprox_conduit_volume")

        window["x"] = window["norm_h_x"]  # * Requirement from PyG
        window["y"] = window["norm_h_y"]  # * Requirement from PyG

        return window

    def normalize_window_eval(self, window):
        head_attributes = [
            "h_x",
            "elevation",
            "ground_level",
        ]  # 'in_offset', 'out_offset'
        flow_attributes = ["q_x"]

        for atr in head_attributes:
            window = self.depths_normalizer(window, atr)

        for atr in flow_attributes:
            if self.norm_flows:
                window[atr] = abs(
                    window[atr]
                )  # ! This is done to avoid the direction of the flow given as sign
            window = self.flows_normalizer.scale_attribute(window, atr)

        window = self.length_normalizer(window, "length")
        window = self.geom_1_normalizer(window, "geom_1")
        window = self.runoff_normalizer(window, "runoff")
        window = self.slope_normalizer(window, "slope")
        window = self.volume_normalizer(window, "aprox_conduit_volume")

        window["x"] = window["norm_h_x"]  # * Requirement from PyG

        return window

    def get_dataloader(self, batch_size):
        list_of_windows = self.get_list_normalized_training_windows()
        return DataLoader(list_of_windows, batch_size)

    def get_unnormalized_heads_pd(self, tensor_heads):
        unnormalized_heads_tensor = self.unnormalize_heads(tensor_heads)
        unnormalized_heads_np = unnormalized_heads_tensor.cpu().detach().numpy()
        unnormalized_heads_pd = pd.DataFrame(
            dict(zip(self.name_nodes, unnormalized_heads_np))
        )
        return unnormalized_heads_pd

    def unnormalize_heads(self, normalized_heads):
        return self.depths_normalizer.unnormalize_attribute(normalized_heads)

    def unscale_flows(self, scaled_flows):
        return self.flows_normalizer.unscale_attribute(scaled_flows)

    def get_unscaled_flows_pd(self, tensor_flows):
        unscaled_flows_tensor = self.unscale_flows(tensor_flows)
        unscaled_flows_np = unscaled_flows_tensor.cpu().detach().numpy()
        unscaled_flows_pd = pd.DataFrame(
            dict(zip(self.name_conduits, unscaled_flows_np))
        )
        return unscaled_flows_pd

    class _InternalNormalizer:
        """
        This internal class is used to normalize the attributes of the training windows.
        Each attribute uses a different InternalNormalizer instance. That way, the normalization is done independently for each attribute.
        Also, the normalizer can track the values used for the normalization process; this is useful for unnormalizing the data.
        """

        def __init__(self, parent, attributes):
            self.parent = parent
            self.attributes = attributes

            maxima = torch.zeros(len(self.attributes))
            minima = torch.zeros(len(self.attributes))
            for i, attribute in enumerate(attributes):
                maxima[i] = self.parent._use_function_get_value(torch.max, attribute)
                minima[i] = self.parent._use_function_get_value(torch.min, attribute)

            self.max_attribute = torch.max(maxima)
            self.min_attribute = torch.min(minima)

        def normalize_attribute(self, window, attribute):
            norm_attribute = self.min_max_normalize(window[attribute])
            window["norm_" + attribute] = norm_attribute.reshape(
                norm_attribute.size()[0], -1
            )
            return window

        def scale_attribute(self, window, attribute):
            norm_attribute = self.min_max_scale(window[attribute])
            window["norm_" + attribute] = norm_attribute.reshape(
                norm_attribute.size()[0], -1
            )
            return window

        def min_max_normalize(self, original_attribute):
            return (original_attribute - self.min_attribute) / (
                self.max_attribute - self.min_attribute
            )

        def unnormalize_attribute(self, attribute):
            return (attribute) * (
                self.max_attribute - self.min_attribute
            ) + self.min_attribute

        def min_max_scale(self, original_attribute):
            return (original_attribute) / (self.max_attribute - self.min_attribute)

        def unscale_attribute(self, original_attribute):
            return (original_attribute) * (self.max_attribute - self.min_attribute)

        def __call__(self, window, attribute):
            return self.normalize_attribute(window, attribute)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"

    def _use_function_get_value(self, f, attribute):
        window = self.training_windows[0]
        extreme = f(window[attribute])

        for window in self.training_windows:
            candidate = f(window[attribute])
            extreme = f(extreme, candidate)
        return extreme

    def __call__(self, window):
        return self.normalize_window(window)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
