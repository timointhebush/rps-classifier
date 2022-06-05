from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from classification.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)  # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()  # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()

        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device)

        # Take feed-forward
        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:  # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict())  # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer():

    def __init__(self, config):
        self.config = config
        self.validation_engine = None
        self.best_model = None

    def train(
            self,
            model, crit, optimizer,
            train_loader, valid_loader
    ):
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        self.validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            self.validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            run_validation,  # function
            self.validation_engine, valid_loader,  # arguments
        )
        self.validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            MyEngine.check_best,  # function
        )
        self.validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            MyEngine.save_model,  # function
            train_engine, self.config,  # arguments
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(self.validation_engine.best_model)
        self.best_model = model
        return model


    def test(self, test_loader):
        print("====== PREDICT TEST DATA ======")
        print("Test:", len(test_loader.dataset))
        self.best_model.eval()
        with torch.no_grad():
            accuracy_sum = 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.validation_engine.device), y.to(self.validation_engine.device)

                y_hat = self.best_model(x)
                loss = self.validation_engine.crit(y_hat, y)
                if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                    accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
                else:
                    accuracy = 0
                print(f'{i + 1}/{len(test_loader)} - loss: {loss}, accuracy: {accuracy}')
                accuracy_sum += accuracy
            accuracy_mean = accuracy_sum / len(test_loader)
        print(f'TEST DATA ACCURACY: {accuracy_mean}')
        return accuracy_mean

