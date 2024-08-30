"""
This module provides the SWMMSimulation class for handling simulation data and extracting windows of data for machine learning models.

Classes:
    SWMMSimulation: A class to manage and process SWMM simulation data.

Usage:
    simulation = SWMMSimulation(G, raw_data, name_simulation)
    one_window = simulation.get_simulation_in_one_window(steps_behind)
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import find
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
import torch


class SWMMSimulation:
    """
    A class to manage and process SWMM simulation data.

    Args:
        G (networkx.Graph): The graph structure of the simulation.
        raw_data (dict): A dictionary containing raw data for heads, flowrate, and runoff.
        name_simulation (str): The name of the simulation.

    Attributes:
        G (networkx.Graph): The graph structure of the simulation.
        heads_raw_data (numpy.ndarray): Raw data for heads.
        flowrate_raw_data (numpy.ndarray): Raw data for flowrate.
        runoff_raw_data (numpy.ndarray): Raw data for runoff.
        name_simulation (str): The name of the simulation.
        simulation_length (int): The length of the simulation data.
    """

    def __init__(self, G, raw_data, name_simulation):

        self.G = G
        self.heads_raw_data = raw_data["heads_raw_data"]
        self.flowrate_raw_data = raw_data["flowrate_raw_data"]
        self.runoff_raw_data = raw_data["runoff_raw_data"]
        # self.rain_raw_data      = raw_data['rain_raw_data']
        self.name_simulation = name_simulation

        self.simulation_length = len(self.runoff_raw_data)

    def get_simulation_in_one_window(self, steps_behind, is_training=True):
        """
        get_simulation_in_one_window method extracts the entire simulation data in one window.

        args:
            steps_behind (int): The number of steps behind the current time.
            is_training (bool): A flag indicating whether the window is for training or testing.
        returns:
            one_window (torch_geometric.data.Data): A window of data accounting for the entire simulation.
        """
        calc_steps_ahead = self.simulation_length - steps_behind
        windows = self.get_all_windows(
            steps_behind=steps_behind,
            steps_ahead=calc_steps_ahead,
            is_training=is_training,
        )
        assert len(windows) == 1, "There should be one and only one window."
        one_window = windows[0]

        return one_window

    def get_all_windows(self, steps_behind, steps_ahead, is_training=True):
        """
        get_all_windows method extracts all the windows of data from a simulation.

        args:
            steps_behind (int): The number of steps behind the current time.
            steps_ahead (int): The number of steps ahead of the current time.
            is_training (bool): A flag indicating whether the window is for training or testing.

        returns:
            windows_list (list): A list of windows of data accounting for the entire simulation.
        """

        assert steps_ahead > 0, "The steps ahead  should be greater than 0"
        assert steps_behind > 0, "The steps behind should be greater than 0"

        length_window = steps_ahead + steps_behind

        max_time_allowed = self.simulation_length - steps_ahead
        windows_list = []

        for time in range(steps_behind - 1, max_time_allowed, length_window):
            if time < self.simulation_length:
                window = self.get_window(
                    steps_behind, steps_ahead, time, is_training=is_training
                )
                windows_list.append(window)
            else:
                break
        return windows_list

    def get_window(self, steps_behind, steps_ahead, time, is_training=True):
        """
        This method extracts a window of data from the simulation data.
        The window includes the data for the given time and the specified number of steps behind and ahead.

        Args:
            steps_behind (int): The number of steps behind the current time.
            steps_ahead (int): The number of steps ahead of the current time.
            time (int): The current time.
            is_training (bool): A flag indicating whether the window is for training or testing.

        Returns:
            window (torch_geometric.data.Data): A window of data for the given time.
        """
        self._checkOutOfBounds(steps_behind, steps_ahead, time)

        h0 = self._get_h0_for_window(time, steps_behind)
        q0 = self._get_q0_for_window(time, steps_behind)
        ro_timeperiod = self._get_ro_for_window(time, steps_ahead, steps_behind)

        h_x_dict = self._get_features_dictionary(h0)
        q_x_dict_conduit = self._get_features_dictionary(q0)
        q_x_dict = self._change_conduit_name_to_edge_tuple(q_x_dict_conduit)
        ro_x_dict = self._get_features_dictionary(ro_timeperiod)

        G_for_window = self.G.to_undirected()

        nx.set_node_attributes(G_for_window, h_x_dict, name="h_x")
        nx.set_node_attributes(G_for_window, ro_x_dict, name="runoff")
        nx.set_edge_attributes(G_for_window, q_x_dict, name="q_x")

        if is_training:
            ht_timeperiod = self._get_ht_for_window(time, steps_ahead)
            qt_timeperiod = self._get_qt_for_window(time, steps_ahead)

            h_y_dict = self._get_features_dictionary(ht_timeperiod)
            q_y_dict_conduit = self._get_features_dictionary(qt_timeperiod)
            q_y_dict = self._change_conduit_name_to_edge_tuple(q_y_dict_conduit)

            nx.set_node_attributes(G_for_window, h_y_dict, name="h_y")
            nx.set_edge_attributes(G_for_window, q_y_dict, name="q_y")

        window = from_networkx(G_for_window)

        window["ground_level"] = window["elevation"] + window["max_depth"]

        geometric_attributes = zip(
            window.geom_1.numpy(), window.geom_2.numpy(), window.length.numpy()
        )
        aprox_volume = torch.tensor(
            list(map(self._get_aprox_volume, geometric_attributes)), dtype=torch.float32
        )
        src, dst = window.edge_index
        slope = abs((window.elevation[dst] - window.elevation[src]) / window.length)

        window["aprox_conduit_volume"] = aprox_volume
        window["slope"] = slope

        window["steps_ahead"] = steps_ahead
        window["steps_behind"] = steps_behind

        return window

    def _get_aprox_volume(self, geoms):
        geom_1, geom_2, length = geoms
        if geom_2 == 0:
            volume = (geom_1**2) * length
        else:
            volume = (geom_1 * geom_2) * length
        return volume

    def _change_conduit_name_to_edge_tuple(self, q_x_dict_conduit):
        return {
            self.G.graph["conduit_phonebook"][k]: value
            for k, value in q_x_dict_conduit.items()
        }

    def _checkOutOfBounds(self, steps_behind, steps_ahead, time):
        max_allowable_time = self.simulation_length - steps_ahead
        if time > max_allowable_time:
            raise AfterEndTimeException
        if time - (steps_behind - 1) < 0:
            raise BeforeZeroTimeException

    def _get_h0_for_window(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        return self.heads_raw_data.iloc[lagged_time : time + 1, :]

    def _get_h0_for_window_tensor(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_h0 = torch.tensor(
            self.heads_raw_data.iloc[lagged_time : time + 1, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_h0

    def _get_q0_for_window(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        return self.flowrate_raw_data.iloc[lagged_time : time + 1, :]

    def _get_q0_for_window_tensor(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_q0 = torch.tensor(
            self.flowrate_raw_data.iloc[lagged_time : time + 1, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_q0

    def _get_ro_for_window(self, time, steps_ahead, steps_behind):
        lagged_time = time - (steps_behind - 1)
        return self.runoff_raw_data.iloc[lagged_time : time + 1 + steps_ahead, :]

    def _get_ro_for_window_tensor(self, time, steps_ahead, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_runoff = torch.tensor(
            self.runoff_raw_data.iloc[lagged_time : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_runoff

    def _get_ht_for_window(self, time, steps_ahead):
        return self.heads_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :]

    def _get_ht_for_window_tensor(self, time, steps_ahead):
        return torch.tensor(
            self.heads_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()

    def _get_qt_for_window(self, time, steps_ahead):
        return self.flowrate_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :]

    def _get_qt_for_window_tensor(self, time, steps_ahead):
        return torch.tensor(
            self.flowrate_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()

    def _get_features_dictionary(self, *args):
        features_df = pd.concat(args).reset_index(drop=True).transpose()
        node_names = list(features_df.index)
        list_features = features_df.values.tolist()
        input_features_dict = dict(zip(node_names, list_features))

        return input_features_dict

    def nx_node_attribute_to_tensor(self, attribute):
        values = list(nx.get_node_attributes(self.G, attribute).values())
        return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def nx_edge_attribute_to_tensor(self, attribute):
        values = list(nx.get_edge_attributes(self.G, attribute).values())
        return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name_simulation})"

    def __len__(self):
        return self.simulation_length


class BeforeZeroTimeException(Exception):
    pass


class AfterEndTimeException(Exception):
    pass
