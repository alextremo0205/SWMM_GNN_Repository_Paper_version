"""
Profiler module.

Useful for estimating the time of computation of a function.
@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import time
import pandas as pd


class Profiler:
    """
    A class for profiling the execution time of a function.

    This class provides a method to profile the execution time of a function by running it multiple times with
    different inputs and recording the execution times. The mean and standard deviation of the execution times
    are computed and stored as attributes of the class.

    Attributes:
        mean_time (float): The mean execution time of the function over all repetitions.
        std_dev (float): The standard deviation of the execution times of the function over all repetitions.

    Methods:
        profile(fc, args, repetitions): Runs the function `fc` with the arguments `args` `repetitions` times and
            records the execution times. Computes the mean and standard deviation of the execution times and stores
            them as attributes of the class.
        plot_times_histogram(): Plots a histogram of the recorded execution times.
    """

    def __init__(self):
        self.mean_time = None
        self.std_dev = None

    def profile(self, fc, args, repetitions):
        recorded_execution_times = []
        self.function = fc
        self.args = args

        for _ in range(repetitions):
            start_time = time.time()

            fc(*args)
            total_time = time.time() - start_time
            recorded_execution_times.append(total_time)

        self.recorded_execution_times = recorded_execution_times
        self._set_pd_stats()

    def _set_pd_stats(self):
        self.pd_times = pd.DataFrame(self.recorded_execution_times)
        self.mean_time = self.pd_times.mean().item()
        self.std_dev = self.pd_times.std().item()

    def plot_times_histogram(self):
        self.pd_times.hist()

    def __str__(self):
        return f"Wall time per repetition: {self.mean_time:.4f} ± {self.std_dev:.4f} seconds"
