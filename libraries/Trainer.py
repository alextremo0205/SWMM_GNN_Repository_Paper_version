"""
This module provides the Trainer classes and a factory function to create trainer instances for training machine learning models.

Classes:
    Trainer (ABC): An abstract base class for trainers.
    Trainer_Heads: A trainer class for training models on head data.

Functions:
    TrainerFactory(trainer_name): Factory function to create trainer instances.

Usage:
    trainer = TrainerFactory("Trainer_Heads")(model, optimizer, criterion, scheduler, report_freq, switch_epoch, min_expected_loss)
    trainer.train(train_loader, val_loader, num_epochs)

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import copy
import time
import wandb
import torch
import datetime
from abc import ABC, abstractmethod


def TrainerFactory(trainer_name):
    available_trainers = {
        "Trainer_Heads": Trainer_Heads,
    }
    return available_trainers[trainer_name]


class Trainer(ABC):
    """
    Trainer class for training machine learning models.
    It is an abstract class that provides the basic structure for training models.
    The class is designed to be inherited by specific trainers for different types of models.
    It provides the following main methods:
    - train(train_loaders, val_loader, epochs): trains the model.
    - get_history(): returns the training and validation losses.
    - set_epoch(epoch): sets the current epoch.

    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        report_freq,
        switch_epoch,
        min_expected_loss,
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.report_freq = report_freq
        self.switch_epoch = switch_epoch

        self.train_losses = []
        self.val_losses = []

        self.min_val_loss = 1e6
        self.best_model_parameters = copy.deepcopy(self.model.state_dict())
        self.last_model_parameters = copy.deepcopy(self.model.state_dict())
        self.epoch_best_model = -1

        self.last_loss = 100
        self.patience = 8
        self.trigger_times = 0

        self.min_expected_loss = min_expected_loss

        self.epoch = 1

    def train(self, train_loaders, val_loader, epochs):
        self._set_initial_state_at_first_epoch(epochs)

        try:
            self._do_training_loop(train_loaders, val_loader, epochs)
        except KeyboardInterrupt:
            pass
        else:
            self._set_final_state_when_finishing(epochs)

    def _set_initial_state_at_first_epoch(self, epochs):
        if self.epoch == 1:
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)
            self._print_initial_message(epochs)
            self.total_time = 0
            self.parameters_before = copy.deepcopy(list(self.model.parameters()))
            self.start_time = time.time()

    def _set_final_state_when_finishing(self, epochs):

        self.total_time = time.time() - self.start_time
        if epochs != 0:
            self.time_per_epoch_sec = self.total_time / epochs
        else:
            self.time_per_epoch_sec = 0
        self._log_wandb_end_values()
        self._print_end_message()

    def get_history(self):
        history = {}
        history["Training loss"] = self.train_losses
        history["Validation loss"] = self.val_losses
        return history

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_best_parameters_in_model(self):
        return self.model.load_state_dict(self.best_model_parameters)

    def _load_last_parameters_in_model(self):
        return self.model.load_state_dict(self.last_model_parameters)

    def _do_training_loop(self, train_loaders, val_loader, epochs):

        while self.epoch <= epochs:
            if self.epoch <= self.switch_epoch:
                train_loader = train_loaders[0]
            else:
                train_loader = train_loaders[1]

            train_loss = self._get_train_loss(train_loader)
            val_loss = self._get_validation_loss(val_loader)

            self.last_model_parameters = copy.deepcopy(self.model.state_dict())
            self._record_best_model(val_loss)

            self.total_time = time.time() - self.start_time
            self.time_per_epoch_sec = self.total_time / self.epoch
            remaining_time = (epochs - self.epoch) * self.time_per_epoch_sec

            self._printCurrentStatus(epochs, train_loss, val_loss, remaining_time)
            wandb.log(
                {
                    "Training loss": train_loss,
                    "Validation loss": val_loss,
                    "epoch": self.epoch,
                }
            )

            if self._should_early_stop(val_loss):
                break
            if self.epoch == 2:
                self._check_parameters_changed_with_training()

            self.epoch += 1

    def _check_parameters_changed_with_training(self):
        parameters_after = copy.deepcopy(list(self.model.parameters()))
        for i in range(len(self.parameters_before)):
            if self._all_entries_in_two_tensors_are_close(
                self.parameters_before[i], parameters_after[i]
            ):
                print("Parameter is not changing: ", i)
                # warnings.warn(f"Parameter {i} is not changing")

    def _record_best_model(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.epoch_best_model = self.epoch
            self.best_model_parameters = copy.deepcopy(self.model.state_dict())

    @abstractmethod
    def _get_train_loss(self, loader):
        pass

    @abstractmethod
    def _get_validation_loss(self, loader):
        pass

    def _should_early_stop(self, val_loss):
        stop_run = False
        if self._is_worth_stopping(val_loss):
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                print("Early stopping!", "The Current Loss:", val_loss)
                stop_run = True
        else:
            self.trigger_times = 0

        self.last_loss = val_loss
        return stop_run

    def _is_worth_stopping(self, current_loss):
        is_loss_stuck = abs(current_loss - self.last_loss) < 1e-4

        is_loss_increasing = current_loss > self.last_loss

        is_loss_exploding = current_loss > 10e5

        is_loss_higher_than_expected = current_loss > self.min_expected_loss

        return (
            is_loss_stuck
            or is_loss_exploding
            or is_loss_increasing
            or is_loss_higher_than_expected
        )

    def _print_initial_message(self, epochs):
        print(
            "train() called:model=%s, trainer=%s, opt=%s(lr=%f), epochs=%d,device=%s\n"
            % (
                type(self.model).__name__,
                type(self).__name__,
                type(self.optimizer).__name__,
                self.optimizer.param_groups[0]["lr"],
                epochs,
                self.device,
            )
        )

    def _printCurrentStatus(self, epochs, train_loss, val_loss, remaining_time):
        epoch = self.epoch
        remaining_time_formatted = str(
            datetime.timedelta(seconds=round(remaining_time))
        )
        if self._is_a_printing_epoch(epochs, epoch):
            print(
                "Epoch %3d/%3d, train loss: %5.3f, val loss: %5.3f, ETA: "
                % (epoch, epochs, train_loss, val_loss)
                + remaining_time_formatted
            )

    def _is_a_printing_epoch(self, epochs, epoch):
        return epoch == 1 or epoch % self.report_freq == 0 or epoch == epochs

    def _print_end_message(self):
        print()
        print("Best model found at epoch: ", self.epoch_best_model)
        print("Best validation loss found: %5.4f" % (self.min_val_loss))
        print()
        print("Time total:     %5.2f sec" % (self.total_time))
        print("Time per epoch: %5.2f sec" % (self.time_per_epoch_sec))

    def _log_wandb_end_values(self):
        wandb.log({"min_val_loss": self.min_val_loss})
        wandb.log({"Training time (s)": self.total_time})
        wandb.log({"Training time per epoch (s)": self.time_per_epoch_sec})

    def _all_entries_in_two_tensors_are_close(self, tensor_a, tensor_b):
        return torch.isclose(tensor_a, tensor_b).all().item()


class Trainer_Heads(Trainer):
    """
    Trainer class for training models on head data.
    It inherits from the Trainer class.
    It provides the following main methods:
    - _get_train_loss(loader): computes the training loss.
    - _get_validation_loss(loader): computes the validation loss.
    """

    def _get_train_loss(self, loader):
        self.model.train()
        train_loss = torch.tensor([0.0], device=self.device)

        for batch in loader:
            x = batch.to(self.device)
            y_nodes = batch.norm_h_y.to(self.device)
            yhat = self.model(x)

            if isinstance(yhat, tuple):
                raise IncorrectTrainerException

            loss = self.criterion(yhat.to(self.device), y_nodes) * x.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss

        train_loss /= len(loader.dataset)

        self.train_losses.append(train_loss.item())

        return train_loss

    def _get_validation_loss(self, loader):
        self.model.eval()
        val_loss = torch.tensor([0.0], device=self.device)
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)
                y_nodes = batch.norm_h_y.to(self.device)
                yhat = self.model(x)

                if isinstance(yhat, tuple):
                    raise IncorrectTrainerException

                loss = self.criterion(yhat.to(self.device), y_nodes) * x.size(0)

                val_loss += loss

        val_loss /= len(loader.dataset)

        self.val_losses.append(val_loss.item())

        return val_loss


class IncorrectTrainerException(Exception):
    pass
