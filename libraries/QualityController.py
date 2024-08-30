"""
This module provides the QualityController class for evaluating the performance of machine learning models.

Classes:
    QualityController: A class to control the quality of model predictions by calculating various metrics.

Usage:
    quality_controller = QualityController(model, normalizer, test_simulations)
    metrics = quality_controller.test_all_simulations_with_metric(metric_state_list)
"""

import time
from libraries.Dashboard import Dashboard
from libraries.MetricCalculator import MetricCalculator


class QualityController:
    def __init__(self, model, normalizer, test_simulations, steps_behind=9):
        self.model = model
        self.normalizer = normalizer
        self.test_simulations = test_simulations
        self._mc = MetricCalculator()

        self.test_windows = [
            test_sim.get_simulation_in_one_window(steps_behind)
            for test_sim in self.test_simulations
        ]
        self.norm_test_windows = [
            normalizer.normalize_window(win) for win in self.test_windows
        ]
        self.G = test_simulations[0].G
        self.elevation = self.test_windows[0].elevation.numpy()

        self._swmm_heads_pd = None
        self._predicted_heads_pd = None
        self.execution_times = {}

    def test_all_simulations_with_metric(self, metric_state_list):
        metric_dict = dict()
        for index, _ in enumerate(self.norm_test_windows):
            simulation_name = self.test_simulations[index].name_simulation
            self.clear_cached_results()
            calculated_metrics = []
            for metric, state in metric_state_list:
                calculated_metrics.append(
                    self.test_one_simulation_with_metric(metric, state, index)
                )
            metric_dict.update({simulation_name: calculated_metrics})
            
        self.clear_cached_results()
        return metric_dict

    def clear_cached_results(self):
        self._swmm_heads_pd = None
        self._predicted_heads_pd = None

    def test_one_simulation_with_metric(self, metric, state, index):
        sim_name = self.test_simulations[index].name_simulation
        norm_window = self.norm_test_windows[index]
        if self._swmm_heads_pd is None or self._predicted_heads_pd is None:
            self._swmm_heads_pd = self.normalizer.get_unnormalized_heads_pd(
                norm_window["y"]
            )
            start_time = time.time()
            y_hat = self.model(norm_window)
            end_time = time.time() - start_time
            self.execution_times.update({sim_name: end_time})
            self._predicted_heads_pd = self.normalizer.get_unnormalized_heads_pd(y_hat)
        self.runoff = norm_window.runoff
        return self._mc.calculate_metric_state(
            metric, self._swmm_heads_pd, self._predicted_heads_pd, self.elevation, state
        )

    def get_flow_percentages(self):
        flow_percentages = {}
        for index, test_sim in enumerate(self.test_simulations):
            sim_name = test_sim.name_simulation
            heads_df = test_sim.heads_raw_data
            elevation = self.test_windows[index].elevation.numpy()

            total_values = heads_df.shape[0] * heads_df.shape[1]
            flow_values = (abs(heads_df - elevation) > 0.1).transpose().sum().sum()
            flow_percentages.update({sim_name: flow_values / total_values})
        return flow_percentages

    def plot_dashboard(self):
        dashboard = Dashboard(
            self._swmm_heads_pd, self._predicted_heads_pd, self.G, self.runoff
        )
        return dashboard.display_results()
