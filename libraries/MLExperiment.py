"""
MLExperiment class is the main class to run the experiments. It is responsible for setting up the experiment, training the model, and evaluating the model. It uses the Trainer class to train the model, the Normalizer class to normalize the data, and the MetricCalculator class to calculate the metrics. It also uses the Profiler class to profile the model and the Dashboard class to display the results.

It receives a configuration dictionary, the data directory, and the saved objects directory as input. It extracts the hyperparameters from the configuration dictionary, defines the data paths, sets up the experiment, and trains the model. It also runs the full experiment, uses the model in validation events, and finishes the ML tracker.

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import sys
from pathlib import Path

sys.path.insert(0, "")

import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import concurrent.futures
import torch.optim as optim
from libraries.Profiler import Profiler
from libraries.Trainer import TrainerFactory
from torch_geometric.loader import DataLoader
from libraries.models.Model_main import ModelFactory
from libraries.MetricCalculator import MetricCalculator
from libraries.Normalizer import NormalizerFactory

import libraries.utils as utils
from libraries.Dashboard import Dashboard


class MLExperiment:
    def __init__(self, config, data_dir, saved_objects_dir):
        """
        Initialize the MLExperiment class.

        Args:
            config (dict): Configuration settings for the experiment.
            data_dir (str): Directory where the data is stored.
            saved_objects_dir (str): Directory where saved objects (models, results) will be stored.

        Attributes:
            device (torch.device): The device to run the experiment on (CPU or GPU).
            config (dict): Configuration settings for the experiment.
            data_dir (str): Directory where the data is stored.
            saved_objects_dir (str): Directory where saved objects (models, results) will be stored.
            metrics_calculator (MetricCalculator): Instance of MetricCalculator for calculating metrics.
            nodes_dashboard (Dashboard or None): Dashboard instance for visualizing nodes, if applicable.
        """

        print("Initializing experiment...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.data_dir = data_dir
        self.saved_objects_dir = saved_objects_dir
        self._extract_hyperparams_from_config()
        self._define_data_paths()
        self._set_up_experiment()
        self.metrics_calculator = MetricCalculator()
        self.nodes_dashboard = None
        print("The experiment is ready")

    def _set_up_experiment(self):
        self._set_up_seeds()
        self._set_up_simulations()
        self._set_up_windows()
        self._set_up_normalizer()
        self._set_up_normalized_windows()
        self._set_up_dataloaders()
        self._set_up_model()
        self._set_up_profiler()
        self._set_up_trainer()

    def train_model(self):
        self.trainer.train(self.training_loaders, self.validation_loader, self.epochs)

    def run_full_experiment(self):
        self._profile_model_with_multiple_events()
        self.train_model()
        self.use_model_in_validation_events()
        self.finish_ML_tracker()

    def use_model_in_validation_events(self):
        val_event_index = 1  # 9
        val_event = self.validation_simulations[val_event_index]
        sim_in_window = val_event.get_simulation_in_one_window(self.steps_behind)
        norm_sim_in_window = self.normalizer.normalize_window(sim_in_window).to(
            self.device
        )
        name_event = val_event.name_simulation

        predicted_heads_pd = self.normalizer.get_unnormalized_heads_pd(
            self.model(norm_sim_in_window)
        )

        swmm_heads_pd = self.normalizer.get_unnormalized_heads_pd(
            norm_sim_in_window["norm_h_y"]
        )

        nodes_to_graph = self.nodes_to_plot
        for node in nodes_to_graph:
            self._save_response_graphs_in_ML_tracker(
                swmm_heads_pd,
                predicted_heads_pd,
                name_event,
                node,
                type_variable="Head",
            )

        if wandb.run is not None:
            elevation = self.validation_windows[0].elevation.numpy()
            wandb.log(
                {
                    "R2 median": round(
                        utils.r2_median(swmm_heads_pd, predicted_heads_pd), 4
                    )
                }
            )
            wandb.log(
                {
                    "R2 WWF": round(
                        utils.r2_median_wet_weather(
                            swmm_heads_pd, predicted_heads_pd, elevation
                        ),
                        4,
                    )
                }
            )
            wandb.log(
                {
                    "R2 DWF": round(
                        utils.r2_median_dry_weather(
                            swmm_heads_pd, predicted_heads_pd, elevation
                        ),
                        4,
                    )
                }
            )

    def change_model(self, new_model):
        self.model = new_model
        self._set_up_trainer()

    def change_normalizer(self, new_normalizer):
        self.normalizer = new_normalizer

        self.normalized_training_windows = [
            [self.normalizer.normalize_window(tra_win) for tra_win in tra_level]
            for tra_level in self.training_windows
        ]
        self.normalized_validation_windows = [
            self.normalizer.normalize_window(val_win)
            for val_win in self.validation_windows
        ]

    @utils.print_function_name
    def finish_ML_tracker(self):
        wandb.join()
        wandb.finish()

    def _extract_hyperparams_from_config(self):
        self.trainer_name = self.config["trainer_name"]
        self.node_loss_weight = self.config["node_loss_weight"]
        self.edge_loss_weight = self.config["edge_loss_weight"]
        self.use_pre_trained_weights = self.config["use_pre_trained_weights"]
        self.requires_freezing = self.config["requires_freezing"]

        self.abs_flows = self.config["abs_flows"]  # Formerly norm_flows

        self.num_events_training = self.config["num_events_training"]
        self.num_events_validation = self.config["num_events_validation"]
        self.balance_ratio = self.config["balance_ratio"]
        self.variance_threshold = self.config["variance_threshold"]

        self.edge_input_list = self.config["edge_input_list"].split(", ")
        self.model_name = self.config["model_name"]
        self.n_hidden_layers = self.config["n_hidden_layers"]
        self.non_linearity = self.config["non_linearity"]
        self.normalizer_name = self.config["normalizer_name"]
        self.steps_behind = self.config["steps_behind"]
        self.steps_ahead = self.config["steps_ahead"]
        self.steps_ahead_validation = self.config["steps_ahead_validation"]
        self.prediction_steps = self.config["prediction_steps"]
        self.hidden_dim = self.config["hidden_dim"]
        self.skip_alpha = self.config["skip_alpha"]

        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.weight_decay = self.config["weight_decay"]
        self.gamma_scheduler = self.config["gamma_scheduler"]
        self.gamma_loss = self.config["gamma_loss"]
        self.switch_epoch = self.config["switch_epoch"]
        self.min_expected_loss = self.config["min_expected_loss"]
        self.use_saved_normalizer = self.config["use_saved_normalizer"]
        self.normalizer_name = self.config["normalizer_name"]
        self.saved_normalizer_name = self.config["saved_normalizer_name"]

        self.seed = self.config["seed"]

        self.k_hops = self.config["k_hops"]
        self.eps_gnn = self.config["eps_gnn"]

        self.nodes_to_plot = self.config["nodes_to_plot"]

    def _define_data_paths(self):
        self.training_simulations_path = (
            Path(self.data_dir) / self.config["network"] / "simulations" / "training"
        )
        self.validation_simulations_path = (
            Path(self.data_dir) / self.config["network"] / "simulations" / "validation"
        )
        self.inp_path = (
            Path(self.data_dir)
            / self.config["network"]
            / "networks"
            / "".join([self.config["network"], ".inp"])
        )
        self.pre_trained_weights_path = (
            Path(self.saved_objects_dir)
            / "saved_model_weights"
            / self.config["pre_trained_weights"]
        )

    @utils.print_function_name
    def _set_up_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed + 1)
        torch.manual_seed(self.seed + 2)
        # torch.use_deterministic_algorithms(True, warn_only = True)

    @utils.print_function_name
    def _set_up_simulations(self):
        self.training_simulations, self.validation_simulations = (
            self._read_simulation_data()
        )

    @utils.print_function_name
    def _set_up_windows(self):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                t1 = executor.submit(self._get_training_windows)
                t2 = executor.submit(self._get_validation_windows)
                self.training_windows = t1.result()
                self.validation_windows = t2.result()

        except Exception as e:
            self.training_windows = self._get_training_windows()
            self.validation_windows = self._get_validation_windows()

        if wandb.run is not None:
            for i, training_windows in enumerate(self.training_windows):
                wandb.log(
                    {"Number of training windows " + str(i): len(training_windows)}
                )
            wandb.log({"Number of validation windows": len(self.validation_windows)})

        # self._free_memory_from_training_simulations()

    @utils.print_function_name
    def _set_up_normalizer(self):
        if self.use_saved_normalizer:
            self.normalizer = utils.load_pickle(
                Path(self.saved_objects_dir)
                / "saved_normalizers"
                / self.saved_normalizer_name
            )
            print("Using saved normalizer: ", self.saved_normalizer_name)

            self.normalizer.name_nodes = self.validation_windows[0].name_nodes

        else:
            self.normalizer = NormalizerFactory(self.normalizer_name)(
                self.training_windows[0], abs_flows=self.abs_flows
            )

    @utils.print_function_name
    def _set_up_normalized_windows(self):

        self.normalized_training_windows = [
            [self.normalizer.normalize_window(tra_win) for tra_win in tra_level]
            for tra_level in self.training_windows
        ]
        self.normalized_validation_windows = [
            self.normalizer.normalize_window(val_win)
            for val_win in self.validation_windows
        ]

    def _save_response_graphs_in_ML_tracker(
        self, swmm_pd, predicted_variable_pd, name_event, item, type_variable="Head"
    ):
        if type_variable == "Head":
            key_wandb = "Head timeseries at node " + item
            title = "Event: " + name_event + "- Head comparison at node " + item
        elif type_variable == "Flow":
            key_wandb = "Flow timeseries at pipe " + item
            title = "Event: " + name_event + "- Flow comparison at pipe " + item

        wandb.log(
            {
                key_wandb: wandb.plot.line_series(
                    xs=[swmm_pd.index.values, predicted_variable_pd.index.values],
                    ys=[swmm_pd[item], predicted_variable_pd[item]],
                    keys=["SWMM", "GNN"],
                    title=title,
                    xname="Time steps",
                )
            }
        )

    @utils.print_function_name
    def _set_up_dataloaders(self):
        self.training_loaders = [
            DataLoader(
                norm_tra_level, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
            for norm_tra_level in self.normalized_training_windows
        ]
        self.validation_loader = DataLoader(
            self.normalized_validation_windows,
            batch_size=self.batch_size,
            drop_last=True,
        )

    @utils.print_function_name
    def _set_up_model(self):
        self.model = ModelFactory(self.model_name)(
            prediction_steps=self.prediction_steps,
            steps_behind=self.steps_behind,
            hidden_dim=self.hidden_dim,
            skip_alpha=self.skip_alpha,
            k_hops=self.k_hops,
            eps_gnn=self.eps_gnn,
            n_hidden_layers=self.n_hidden_layers,
            non_linearity=self.non_linearity,
            edge_input_list=self.edge_input_list,
        )
        self.model.to(self.device)

        self.assess_pre_training_weights()

        if wandb.run is not None:
            wandb.log({"Number of trainable parameters": self.model.count_parameters()})

    def assess_pre_training_weights(self):
        if self.use_pre_trained_weights:
            saved_weights_dict = torch.load(
                self.pre_trained_weights_path, map_location="cpu"
            )
            self.model.load_state_dict(saved_weights_dict, strict=False)
            print("Using pre-trained weights:", self.config["pre_trained_weights"])
            if self.requires_freezing:
                for parameter_name, param in self.model.named_parameters():
                    if parameter_name in saved_weights_dict.keys():
                        param.requires_grad = False

    @utils.print_function_name
    def _set_up_profiler(self):
        self.profiler = Profiler()

    @utils.print_function_name
    def _profile_model_with_multiple_events(self):

        names_sim = ["synt_81", "real_91", "real_87"]
        simulation_times = [5, 20, 50]
        check_sim_name = lambda sim: sim.name_simulation in names_sim
        simulations = list(filter(check_sim_name, self.validation_simulations))

        for i in range(len(simulations)):
            self._profile_simulation(simulations[i], simulation_times[i])

    def _profile_simulation(self, simulation, simulation_time):
        event = simulation.get_simulation_in_one_window(self.steps_behind)
        normalized_event = self.normalizer.normalize_window(event)
        with torch.no_grad():
            self.profiler.profile(
                self.model, [normalized_event.to(self.device)], repetitions=5
            )
        average_time = self.profiler.mean_time
        name = simulation.name_simulation
        wandb.log({"Average time in " + name: average_time})
        wandb.log({"Standard deviation time in " + name: self.profiler.std_dev})
        wandb.log(
            {
                "Rough speed up (vs. "
                + str(simulation_time)
                + " sec) "
                + name: simulation_time / average_time
            }
        )

    @utils.print_function_name
    def _set_up_trainer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.gamma_scheduler
        )  # step_size=30,
        self.criterion = nn.MSELoss()

        chosen_trainer = TrainerFactory(self.trainer_name)
        self.trainer = chosen_trainer(
            self.model,
            self.optimizer,
            self.criterion,
            self.scheduler,
            report_freq=2,
            switch_epoch=self.switch_epoch,
            min_expected_loss=self.min_expected_loss,
            node_loss_weight=self.node_loss_weight,
            edge_loss_weight=self.edge_loss_weight,
        )

    def _read_simulation_data(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1 = executor.submit(
                utils.extract_simulations_from_folders,
                self.training_simulations_path,
                self.inp_path,
                self.num_events_training,
            )
            t2 = executor.submit(
                utils.extract_simulations_from_folders,
                self.validation_simulations_path,
                self.inp_path,
                self.num_events_validation,
            )

            training_simulations = t1.result()
            validation_simulations = t2.result()

        return training_simulations, validation_simulations

    def _get_training_windows(self):
        training_windows_names = self.config["training_windows_names"]
        use_saved_training_windows = self.config["use_saved_training_windows"]

        folder_path = (
            Path(self.saved_objects_dir) / "saved_windows" / self.config["network"]
        )
        if use_saved_training_windows:
            training_windows = []
            for path in training_windows_names:
                training_windows.append(utils.load_pickle(folder_path / path))
                print("Using loaded windows: ", path)

        else:
            training_windows = []
            for step_ahead in self.steps_ahead:
                instance_windows = self._get_balanced_windows_from_list_simulations(
                    self.training_simulations, self.steps_behind, step_ahead
                )
                training_windows.append(instance_windows)

        return training_windows

    def _get_validation_windows(self):
        validation_windows_name = self.config["validation_windows_name"]
        use_saved_validation_windows = self.config["use_saved_validation_windows"]

        folder_path = (
            Path(self.saved_objects_dir) / "saved_windows" / self.config["network"]
        )
        validation_windows_path = folder_path / validation_windows_name

        if use_saved_validation_windows:
            validation_windows = utils.load_pickle(validation_windows_path)
            print("Using loaded windows: ", validation_windows_name)
        else:
            validation_windows = self._get_balanced_windows_from_list_simulations(
                self.validation_simulations,
                self.steps_behind,
                self.steps_ahead_validation,
            )
        return validation_windows

    def _get_balanced_windows_from_list_simulations(
        self, list_simulations, steps_behind, steps_ahead
    ):
        windows_list = []
        static_window = []

        for sim in list_simulations:
            heads = sim.heads_raw_data
            simulation_length = len(heads)
            length_window = steps_ahead + steps_behind
            max_time_allowed = simulation_length - steps_ahead

            t = steps_behind - 1
            while t < max_time_allowed:
                if (
                    heads.iloc[t : t + length_window].var().max()
                    > self.variance_threshold
                ) and t < simulation_length:
                    window = sim.get_window(steps_behind, steps_ahead, t)
                    windows_list.append(window)

                    t += length_window
                else:
                    if len(static_window) == 0:
                        window = sim.get_window(steps_behind, steps_ahead, t)
                        static_window.append(window)
                    t += 1

        windows = windows_list + static_window * (
            self.balance_ratio * len(windows_list)
        )
        random.shuffle(windows)

        return windows

    def run_model_in_validation_event(self, event_index=5):
        val_event = self.validation_simulations[event_index]
        normalizer = self.normalizer
        steps_behind = self.steps_behind

        sim_in_window = val_event.get_simulation_in_one_window(steps_behind)

        norm_sim_in_window = normalizer.normalize_window(sim_in_window)

        yhat = self.model(norm_sim_in_window.to(self.device))

        y_heads = yhat

        target_heads = norm_sim_in_window["norm_h_y"]

        self.swmm_heads_pd = normalizer.get_unnormalized_heads_pd(target_heads)
        self.predicted_heads_pd = normalizer.get_unnormalized_heads_pd(y_heads)
        self.runoff = val_event.runoff_raw_data

    def run_traced_model_in_validation_event(self, event_index=5):
        val_event = self.validation_simulations[event_index]
        normalizer = self.normalizer
        steps_behind = self.steps_behind

        sim_in_window = val_event.get_simulation_in_one_window(steps_behind)

        norm_sim_in_window = normalizer.normalize_window(sim_in_window)

        input_list = [
            norm_sim_in_window.norm_h_x,
            norm_sim_in_window.edge_index,
            norm_sim_in_window.norm_ground_level,
            norm_sim_in_window.norm_runoff,
            norm_sim_in_window.norm_elevation,
            norm_sim_in_window.norm_length,
            norm_sim_in_window.norm_geom_1,
        ]

        with torch.no_grad():
            yhat = self.model(*input_list)

        if isinstance(yhat, tuple):
            y_heads = yhat[0]
        else:
            y_heads = yhat

        target_heads = norm_sim_in_window["norm_h_y"]

        self.swmm_heads_pd = normalizer.get_unnormalized_heads_pd(target_heads)
        self.predicted_heads_pd = normalizer.get_unnormalized_heads_pd(y_heads)
        self.runoff = val_event.runoff_raw_data

    def run_model_in_given_norm_window(self, norm_window):

        yhat = self.model(norm_window.to(self.device))

        y_heads = yhat

        self.swmm_heads_pd = self.normalizer.get_unnormalized_heads_pd(
            norm_window["norm_h_y"]
        )
        self.predicted_heads_pd = self.normalizer.get_unnormalized_heads_pd(y_heads)

    def display_results(self, optional_runoff=None):
        G = self.validation_simulations[0].G
        if optional_runoff is None:
            try:
                self.nodes_dashboard = Dashboard(
                    self.swmm_heads_pd,
                    self.predicted_heads_pd,
                    G,
                    self.runoff.iloc[self.steps_behind :, :],
                )
            except:
                self.nodes_dashboard = Dashboard(
                    self.swmm_heads_pd, self.predicted_heads_pd, G, None
                )

        else:
            self.nodes_dashboard = Dashboard(
                self.swmm_heads_pd, self.predicted_heads_pd, G, optional_runoff
            )
        return self.nodes_dashboard.display_results()

    def get_performance_in_heads(self):
        elevation = self.validation_windows[0].elevation.numpy()

        performance_dict = {
            "Overall": round(
                utils.r2_overall(self.swmm_heads_pd, self.predicted_heads_pd), 4
            ),
            "Flow": round(
                utils.r2_wet_weather(
                    self.swmm_heads_pd, self.predicted_heads_pd, elevation
                ),
                4,
            ),
            "No flow": round(
                utils.r2_dry_weather(
                    self.swmm_heads_pd, self.predicted_heads_pd, elevation
                ),
                4,
            ),
        }

        return performance_dict

    def get_performance_in_flows(self, metric="MSE"):

        performance_dict = {
            "Overall": round(
                self.metrics_calculator.calculate_metric_flows_state(
                    metric, self.swmm_flows_pd, self.predicted_flows_pd, state="overall"
                ),
                6,
            ),
            "Flow": round(
                self.metrics_calculator.calculate_metric_flows_state(
                    metric, self.swmm_flows_pd, self.predicted_flows_pd, state="wet"
                ),
                6,
            ),
            "No flow": round(
                self.metrics_calculator.calculate_metric_flows_state(
                    metric, self.swmm_flows_pd, self.predicted_flows_pd, state="dry"
                ),
                6,
            ),
        }

        return performance_dict

    def calculate_metrics_for_all_validation_events(self):
        metrics = {}
        for i in range(len(self.validation_simulations)):
            self.run_model_in_validation_event(i)
            metrics[i] = {
                "Heads": self.get_performance_in_heads(),
                "Flows": self.get_performance_in_flows(),
            }
        return metrics
