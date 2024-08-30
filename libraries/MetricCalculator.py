"""
The MetricCalculator class is used to calculate different metrics for the evaluation of the model.

The included metrics are:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Coefficient of Determination (COD)
- Error Sum
- Sum Difference

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


class MetricCalculator:
    def __init__(self):
        self.metric_functions = {
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
            "MAPE": mean_absolute_percentage_error,
            "sMAPE": self.symmetric_mean_absolute_percentage_error,
            "COD": self.r2_overall,
            "error_sum": self._error_sum,
            "sum_difference": self._sum_difference,
        }

    def calculate_metric(
        self, metric_name, y_true_matrix, y_pred_matrix, multioutput="raw_values"
    ):
        ans = self.metric_functions[metric_name](
            y_true_matrix, y_pred_matrix, multioutput=multioutput
        )
        return ans

    def calculate_metric_state(
        self,
        metric_name,
        y_true_matrix,
        y_pred_matrix,
        elevation,
        state,
        multioutput="uniform_average",
    ):
        if state == "dry":
            mask = abs(y_true_matrix - elevation) < 0.01  # 0.00001
        elif state == "wet":
            mask = abs(y_true_matrix - elevation) > 0.01  # 0.00001
        elif state == "overall":
            mask = y_true_matrix - y_true_matrix < 100  # trick so that all are trues

        # average_time = (mask.sum()/mask.count()).mean()

        masked_swmm_heads_pd = y_true_matrix[mask]
        masked_predicted_heads_pd = y_pred_matrix[mask]

        masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
        masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

        masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
        masked_predicted_heads = masked_predicted_heads[
            ~np.isnan(masked_predicted_heads)
        ]

        return self.calculate_metric(
            metric_name,
            masked_swmm_heads,
            masked_predicted_heads,
            multioutput=multioutput,
        )

    def calculate_metric_flows_state(
        self,
        metric_name,
        y_true_matrix,
        y_pred_matrix,
        state,
        multioutput="uniform_average",
    ):
        if state == "dry":
            mask = abs(y_true_matrix) < 0.001  # 0.00001
        elif state == "wet":
            mask = abs(y_true_matrix) >= 0.001  # 0.00001
        elif state == "overall":
            mask = y_true_matrix - y_true_matrix < 100  # trick so that all are trues

        # average_time = (mask.sum()/mask.count()).mean()

        masked_swmm_heads_pd = y_true_matrix[mask]
        masked_predicted_heads_pd = y_pred_matrix[mask]

        masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
        masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

        masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
        masked_predicted_heads = masked_predicted_heads[
            ~np.isnan(masked_predicted_heads)
        ]

        return self.calculate_metric(
            metric_name,
            masked_swmm_heads,
            masked_predicted_heads,
            multioutput=multioutput,
        )

    def _error_sum(self, y_true_matrix, y_pred_mat, multioutput="raw_values"):
        ans = None
        if multioutput == "raw_values":
            ans = np.array((y_pred_mat - y_true_matrix).sum())
        elif multioutput == "uniform_average":
            ans = ans = np.array((y_pred_mat - y_true_matrix).sum().mean())
        return ans

    def _sum_difference(self, y_true_matrix, y_pred_mat, multioutput="raw_values"):
        ans = None
        if multioutput == "raw_values":
            ans = np.array((y_pred_mat.sum() - y_true_matrix.sum()))
        elif multioutput == "uniform_average":
            ans = ans = np.array((y_pred_mat.sum() - y_true_matrix.sum()).mean())
        return ans

    def r2_overall(self, swmm_variable, predicted_variable, *args, **kwargs):
        flattened_swmm = swmm_variable.flatten()
        flattened_prediction = predicted_variable.flatten()
        return r2_score(flattened_swmm, flattened_prediction)

    def symmetric_mean_absolute_percentage_error(self, A, F):
        return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
