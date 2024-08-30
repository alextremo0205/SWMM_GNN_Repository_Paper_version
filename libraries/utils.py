"""
This module provides utility functions for various tasks such as loading YAML files, reading text files, and handling data.

Functions:
    print_function_name(fn): A decorator to print the name of the function being executed.
    load_yaml(yaml_path): Loads a YAML file from the given path.
    get_lines_from_textfile(path): Reads all lines from a text file.
    get_info_from_file(path): Reads a CSV file into a pandas DataFrame.

Usage:
    @print_function_name
    def example_function():
        pass

    yaml_data = load_yaml("config.yaml")
    lines = get_lines_from_textfile("example.txt")
    info = get_info_from_file("data.csv")
"""

import os
import yaml
from yaml.loader import SafeLoader

import pickle
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import r2_score
from libraries.SWMM_Simulation import SWMMSimulation
from libraries.SWMM_Converter import SWMM_Converter


def print_function_name(fn):
    """
    A decorator to print the name of the function being executed.

    Args:
        fn (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    def inner(*args, **kwargs):
        print("{0} executing...".format(fn.__name__))
        to_execute = fn(*args, **kwargs)
        return to_execute

    return inner


def load_yaml(yaml_path):
    """
    Loads a YAML file from the given path.

    Args:
        yaml_path (str): The path to the YAML file.

    Returns:
        dict: The loaded YAML data.

    Raises:
        InvalidYAMLPathException: If the YAML file does not exist.
    """
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            yaml_data = yaml.load(f, Loader=SafeLoader)
    else:
        raise InvalidYAMLPathException
    return yaml_data


def get_lines_from_textfile(path):
    """
    Reads all lines from a text file.

    Args:
        path (str): The path to the text file.

    Returns:
        list: A list of lines from the text file.
    """
    with open(path, "r") as fh:
        lines = fh.readlines()
    return lines


def get_info_from_file(path):
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    info = pd.read_csv(path)
    return info


def get_rain_in_pandas(rain_path):
    rainfall_raw_data = pd.read_csv(rain_path, sep="\t", header=None)
    rainfall_raw_data.columns = [
        "station",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "value",
    ]
    rainfall_raw_data = rainfall_raw_data[
        :-1
    ]  # Drop last row to syncronize it with heads
    return rainfall_raw_data


def get_heads_from_file(path):
    """
    Reads head data from a CSV file and processes the column names.

    Args:
        path (str): The path to the CSV file containing head data.

    Returns:
        pandas.DataFrame: The processed head data.
    """
    head_raw_data = pd.read_csv(path, index_col=0)  # get_info_from_file(path)
    head_raw_data.columns = head_raw_data.columns.str.replace("_Hydraulic_head", "")
    head_raw_data.columns = head_raw_data.columns.str.replace("node_", "")
    return head_raw_data


def get_flowrate_from_file(path):
    """
    Reads flowrate data from a CSV file and processes the column names.

    Args:
        path (str): The path to the CSV file containing flowrate data.

    Returns:
        pandas.DataFrame: The processed flowrate data.
    """
    flowrate_raw_data = pd.read_csv(path, index_col=0)
    flowrate_raw_data.columns = flowrate_raw_data.columns.str.replace("_Flow_rate", "")
    flowrate_raw_data.columns = flowrate_raw_data.columns.str.replace("link_", "")
    return flowrate_raw_data


def get_runoff_from_file(path):
    """
    Reads runoff data from a CSV file and processes the column names.

    Args:
        path (str): The path to the CSV file containing runoff data.

    Returns:
        pandas.DataFrame: The processed runoff data.
    """

    # ! This function assumes the naming convention of the runoff files.
    # * Before, it assumed that all nodes were connected to a subcatchment and they shared the same name.

    runoff_raw_data = pd.read_csv(path, index_col=0)
    runoff_raw_data.columns = runoff_raw_data.columns.str.replace("_Runoff_rate", "")
    runoff_raw_data.columns = runoff_raw_data.columns.str.replace("subcatchment_", "")
    return runoff_raw_data


def get_dry_periods_index(rainfall_raw_data):
    """
    Identifies dry periods in the rainfall data.

    Args:
        rainfall_raw_data (pandas.DataFrame): The rainfall data with a 'value' column indicating rainfall amounts.

    Returns:
        list: A list of lists, where each sublist contains the indexes of a single dry period.
    """
    indexes = np.array(rainfall_raw_data[rainfall_raw_data["value"] == 0].index)
    differences = np.diff(indexes)

    dry_periods_index = []
    single_dry_period_indexes = []
    for i, j in enumerate(differences):
        if j == 1:
            single_dry_period_indexes.append(i)
        else:
            dry_periods_index.append(single_dry_period_indexes)
            single_dry_period_indexes = []

    return dry_periods_index


def extract_simulations_from_folders(simulations_path, inp_path, max_events=-1):
    """
    Extracts simulation data from folders and creates SWMMSimulation instances.

    Args:
        simulations_path (str or Path): The path to the directory containing simulation folders.
        inp_path (str or Path): The path to the SWMM input file (.inp).
        max_events (int, optional): The maximum number of simulation events to extract. Defaults to -1, which means all events.

    Returns:
        list: A list of SWMMSimulation instances containing the extracted simulation data.
    """
    list_of_simulations = os.listdir(simulations_path)

    if max_events == -1:
        max_events = len(list_of_simulations)

    converter = SWMM_Converter(inp_path, is_directed=True)
    G = converter.inp_to_G()

    simulations = []

    num_saved_events = 0
    for name_simulation in list_of_simulations:
        hydraulic_heads_path = simulations_path / name_simulation / "hydraulic_head.csv"
        flowrate_raw_path = simulations_path / name_simulation / "flow_rate.csv"
        runoff_path = simulations_path / name_simulation / "runoff.csv"
        # rain_path = simulations_path / name_simulation / name_simulation+'.dat'

        heads_raw_data = get_heads_from_file(hydraulic_heads_path)
        flowrate_raw_data = get_flowrate_from_file(flowrate_raw_path)
        runoff_raw_data = get_runoff_from_file(runoff_path)

        nodes_subcatchment = nx.get_node_attributes(G, "subcatchment")

        reversed_nodes_subcatchment = {
            value: key for key, value in nodes_subcatchment.items()
        }

        runoff_raw_data = runoff_raw_data.rename(columns=reversed_nodes_subcatchment)
        missing_nodes = set(heads_raw_data.columns) - set(runoff_raw_data.columns)

        for i in missing_nodes:
            runoff_raw_data[i] = 0

        # rain_raw_data = get_rain_in_pandas(rain_path)
        raw_data = {
            "heads_raw_data": heads_raw_data,
            "flowrate_raw_data": flowrate_raw_data,
            "runoff_raw_data": runoff_raw_data,
        }
        # "rain_raw_data":rain_raw_data}
        sim = SWMMSimulation(G, raw_data, name_simulation)
        simulations.append(sim)

        if num_saved_events >= max_events:
            break

        num_saved_events += 1
    return simulations


def get_all_windows_from_list_simulations(simulations, steps_behind, steps_ahead):
    """
    It converts a list of SWMMSimulation instances into a list of windows.

    Args:
        simulations (list): A list of SWMMSimulation instances.
        steps_behind (int): The number of steps behind to consider in the window.
        steps_ahead (int): The number of steps ahead to consider in the window.
    Returns:
        list: A list of windows.

    """

    windows = []
    for sim in simulations:
        windows += sim.get_all_windows(
            steps_behind=steps_behind, steps_ahead=steps_ahead
        )
    return windows


def save_pickle(variable, path):
    with open(path, "wb") as handle:
        pickle.dump(variable, handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        variable = pickle.load(handle)
    return variable


def run_model_in_validation_event(ml_experiment, model, event_index=5):
    """
    Executes the model in a validation event and returns the SWMM heads and the predicted heads.
    args:
        ml_experiment (MLExperiment): The MLExperiment instance.
        model (PyTorch model): The model to be executed.
        event_index (int): The index of the event to be executed.

    Returns:
        pandas.DataFrame: The SWMM heads.
        pandas.DataFrame: The predicted heads.
    """

    val_event = ml_experiment.validation_simulations[event_index]
    normalizer = ml_experiment.normalizer
    steps_behind = ml_experiment.steps_behind

    sim_in_window = val_event.get_simulation_in_one_window(steps_behind)
    norm_sim_in_window = normalizer.normalize_window(sim_in_window)

    swmm_heads_pd = normalizer.get_unnormalized_heads_pd(norm_sim_in_window["norm_h_y"])

    predicted_heads_pd = normalizer.get_unnormalized_heads_pd(model(norm_sim_in_window))
    return swmm_heads_pd, predicted_heads_pd


def r2_median_wet_weather(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the median R2 score for wet weather conditions.
    """

    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_median(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_median_dry_weather(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the median R2 score for dry weather conditions.
    """
    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_median(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_wet_weather(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the R2 score for wet weather conditions.
    """
    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_dry_weather(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the R2 score for dry weather conditions.
    """
    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_flow_wet_weather(swmm_flows_pd, predicted_heads_pd):
    """
    It calculates the R2 score for wet weather conditions.
    """
    mask_wet = abs(swmm_flows_pd) > 0.00001

    masked_swmm_heads_pd = swmm_flows_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(masked_swmm_heads, masked_predicted_heads)


def r2_flow_dry_weather(swmm_flows_pd, predicted_flows_pd):
    """
    It calculates the R2 score for dry weather conditions.
    """
    mask_dry = abs(swmm_flows_pd) < 0.00001

    masked_swmm_flows_pd = swmm_flows_pd[mask_dry]
    masked_predicted_flows_pd = predicted_flows_pd[mask_dry]

    masked_swmm_flows = masked_swmm_flows_pd.to_numpy()
    masked_swmm_flows = masked_swmm_flows[~np.isnan(masked_swmm_flows)]

    masked_predicted_flows = masked_predicted_flows_pd.to_numpy()
    masked_predicted_flows = masked_predicted_flows[~np.isnan(masked_predicted_flows)]

    return r2_score(masked_swmm_flows, masked_predicted_flows)


def r2_overall(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):
    """
    It calculates the overall R2 score a simulation
    """
    flattened_swmm = swmm_variable_pd.to_numpy().flatten()
    flattened_prediction = predicted_variable_pd.to_numpy().flatten()
    return r2_score(flattened_swmm, flattened_prediction)


def wet_r2_per_node(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the R2 score for wet weather conditions per node.
    """

    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    wet_r2 = []
    for i in range(len(masked_swmm_heads_pd.columns)):
        try:
            r2 = r2_score(
                masked_swmm_heads_pd.iloc[:, i].dropna().to_numpy(),
                masked_predicted_heads_pd.iloc[:, i].dropna().to_numpy(),
            )
        except Exception as e:
            r2 = -9.99
        r2 = np.clip(r2, -10, 1)
        wet_r2.append(r2)
    return wet_r2


def dry_r2_per_node(swmm_heads_pd, predicted_heads_pd, elevation):
    """
    It calculates the R2 score for dry weather conditions per node.
    """
    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    dry_r2 = []
    for i in range(len(masked_swmm_heads_pd.columns)):
        try:
            r2 = r2_score(
                masked_swmm_heads_pd.iloc[:, i].dropna().to_numpy(),
                masked_predicted_heads_pd.iloc[:, i].dropna().to_numpy(),
            )
        except Exception as e:
            r2 = -9.99

        r2 = np.clip(r2, -10, 1)
        dry_r2.append(r2)

    return dry_r2


def r2_median(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):
    """
    It calculates the median R2 score.
    """
    r2s = []
    for i in range(len(swmm_variable_pd.columns)):
        r2s.append(
            r2_score(
                swmm_variable_pd.iloc[:, i].to_numpy(),
                predicted_variable_pd.iloc[:, i].to_numpy(),
            )
        )
    return np.median(r2s)


def r2_mean(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):
    """
    It calculates the mean of the R2 score.
    """

    r2s = []
    for i in range(len(swmm_variable_pd.columns)):
        r2s.append(
            r2_score(
                swmm_variable_pd.iloc[:, i].to_numpy(),
                predicted_variable_pd.iloc[:, i].to_numpy(),
            )
        )
    return np.mean(r2s)


class InvalidYAMLPathException(Exception):
    pass
