from typing import Any, Dict, Tuple
from abc import abstractmethod

import pandas as pd
import torch
from torch import nn
from omegaconf import DictConfig
from lightning import LightningModule

from collections import defaultdict


class BaseLitModule(LightningModule):
    """Base LightningModule for all model types.

    Handles optimizer/scheduler configuration, metric logging, checkpoint
    cleaning, and per-sample output storage. Subclasses must implement
    ``model_step``.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        metrics: Dict[str, torch.nn.Module],
        metrics_config: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        checkpoint: torch.nn.Module = None,
        checkpoint_type: str = 'lit_module',
        generation: DictConfig = None,
        save_outputs: dict = None,
        watch: DictConfig = None,
        **kwargs
    ) -> None:
        """Initialize BaseLitModule with model, loss, metrics, optimizer, and scheduler."""
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'criterion', 'metrics'])
        self.sync_dist = True
        self.net = net
        self.criterion = criterion
        self.metrics = nn.ModuleDict(metrics)
        self.metrics_config = metrics_config

        self.log_kwargs = {
            'train':  {'on_step': True, 'on_epoch': True, 'prog_bar': True},
            'val': {'on_step': True, 'on_epoch': True, 'prog_bar': True},
            'test': {'on_step': True, 'on_epoch': True, 'prog_bar': True},
        }

        self.init_params_for_logging(watch)

        # Output storage collects per-sample data during stages (typically val/test)
        # and saves it to a file at the end of each epoch.
        if save_outputs is not None:
            self.output_storage = dict()

            for stage in ['train', 'val', 'test']:
                if stage in self.hparams.save_outputs:
                    settings = self.hparams.save_outputs[stage]
                    if "fields" not in settings or "file" not in settings:
                        raise ValueError(f"save_outputs['{stage}'] must have 'fields' and 'file' keys")

                    self.output_storage[stage] = defaultdict(list)

    def init_params_for_logging(self, watch):
        """Register model parameters to be logged at each training step.

        :param watch: Config mapping parameter names to module types.
            Each entry specifies a parameter name and a ``func`` field
            indicating the module class to search for.
        """
        self._params_to_log = []
        if watch is not None:
            for name in watch:
                params = self.find_param(name, watch[name].func)

                if len(params) == 0:
                    raise ValueError(f"Parameter `{name}` is not found. If the parameter is in a child module the name "
                                     f"has to contain the child's module name first: `attr_name.{name}`")

                self._params_to_log += [(f"{name}_{i}", param) for i, param in enumerate(params)]

    def find_param(self, name, instance):
        """Find all parameters with the given name inside modules of the specified type.

        :param name: Parameter name to search for (e.g. ``"weight"``).
        :param instance: Module class to filter by (e.g. ``nn.Linear``).
        :return: List of matching parameter tensors.
        """
        all_params = []
        for module in self.net.modules():
            if isinstance(module, instance):
                for param_name, param in module.named_parameters():
                    if param_name == name:
                        all_params.append(param)
        return all_params

    @abstractmethod
    def model_step(
            self,
            stage_name: str,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss, predictions, and targets for a single batch.

        Must be implemented by subclasses.

        :param stage_name: Stage name (``"train"``, ``"val"``, or ``"test"``).
        :param batch: A batch of data from the dataloader.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader (for multiple val/test loaders).
        :return: Tuple of (loss, predictions, targets).
        """
        pass

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        :param batch: A batch of data from the training dataloader.
        :param batch_idx: Index of the current batch.
        :return: Loss or None if loss is unavailable.
        """
        loss, preds, targets = self.model_step('train', batch, batch_idx)

        if loss is None:
            return None

        self.log_step('train', batch, loss, preds, targets)
        self.log_params()
        self.add_to_storage('train', batch)

        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step.

        :param batch: A batch of data from the validation dataloader.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader.
        """
        stage = 'val'
        loss, preds, targets = self.model_step(stage, batch, batch_idx, dataloader_idx=dataloader_idx)

        if loss is None:
            return None

        self.log_step(stage, batch, loss, preds, targets)
        self.add_to_storage(stage, batch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single test step.

        :param batch: A batch of data from the test dataloader.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader.
        """
        stage = 'test'
        loss, preds, targets = self.model_step(stage, batch, batch_idx, dataloader_idx=dataloader_idx)

        if loss is None:
            return None

        self.log_step(stage, batch, loss, preds, targets)
        self.add_to_storage(stage, batch)

    def on_train_epoch_end(self) -> None:
        """Save collected outputs at the end of each training epoch."""
        self.save_outputs()

    def on_validation_epoch_end(self) -> None:
        """Save collected outputs at the end of each validation epoch."""
        self.save_outputs()

    def on_test_epoch_end(self) -> None:
        """Save collected outputs at the end of each test epoch."""
        self.save_outputs()

    def on_save_checkpoint(self, checkpoint):
        """Remove specified keys from the checkpoint before saving."""
        self.clean_checkpoint(checkpoint)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.

        :return: Dict with ``"optimizer"`` and optionally ``"lr_scheduler"`` keys.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler.instance(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler.params.monitor,
                    "interval": self.hparams.scheduler.params.interval,
                    "frequency": self.hparams.scheduler.params.frequency,
                },
            }

        return {"optimizer": optimizer}

    def log_step(self, stage, batch, loss, preds=None, targets=None):
        """Log loss, batch size, and metrics for a single step.

        :param stage: Stage name (``"train"``, ``"val"``, or ``"test"``).
        :param batch: The current batch dict.
        :param loss: Loss value to log.
        :param preds: Model predictions (optional).
        :param targets: Ground truth targets (optional).
        """
        batch_size = None

        if 'batch_size' in batch:
            batch_size = batch['batch_size']
            self.log(f"{stage}/batch_size", batch_size, batch_size=batch_size, sync_dist=self.sync_dist)

        self.log(f"{stage}/loss", loss, **self.log_kwargs[stage], batch_size=batch_size, sync_dist=self.sync_dist)

        if preds is not None and targets is not None:
            self.log_metrics(stage, preds, targets, batch)

    def log_metrics(self, stage, preds, targets, batch):
        """Compute and log all configured metrics for the given stage.

        When per-sample scoring is enabled (via ``save_outputs``), the metric
        returns both an aggregate value and per-sample scores. Per-sample scores
        are stored in the batch dict for later extraction.

        :param stage: Stage name.
        :param preds: Model predictions.
        :param targets: Ground truth targets.
        :param batch: The current batch dict (modified in-place with metric values).
        """
        for metric in self.metrics_config[stage]:
            self.metrics[metric].module_device = self.device
            metric_data = []
            if hasattr(self.metrics[metric], 'required_data'):
                metric_data = self.metric_data_from_batch(batch, self.metrics[metric].required_data)

            score_per_sample = self.is_calc_metric_per_sample(stage, metric)

            if score_per_sample:
                value = self.metrics[metric](preds, targets, *metric_data, score_per_sample=True)
            else:
                value = self.metrics[metric](preds, targets, *metric_data)

            if score_per_sample:
                value, value_per_sample = value
                batch[metric] = value_per_sample
            else:
                batch[metric] = value

            self.log(f"{stage}/{metric}", value, **self.log_kwargs[stage], sync_dist=self.sync_dist)

    def is_calc_metric_per_sample(self, stage, metric_name):
        """Check whether per-sample metric scoring is needed for output storage.

        :param stage: Stage name.
        :param metric_name: Name of the metric.
        :return: True if the metric should produce per-sample scores.
        """
        if (self.hparams.save_outputs is not None
                and stage in self.hparams.save_outputs
                and metric_name in set(self.hparams.save_outputs[stage].fields)):
            return True

        return False

    def add_to_storage(self, stage, batch):
        """Accumulate batch fields into output storage for later saving.

        :param stage: Stage name.
        :param batch: The current batch dict containing fields to store.
        """
        if self.hparams.save_outputs is not None and stage in self.hparams.save_outputs:
            settings = self.hparams.save_outputs[stage]
            for name in settings.fields:
                if isinstance(batch[name], torch.Tensor):
                    values = batch[name].cpu().numpy().tolist()
                else:
                    values = batch[name]
                self.output_storage[stage][name] += values

    def save_outputs(self):
        """Write accumulated output storage to CSV files.

        Called at the end of each epoch. Each stage with configured output
        storage is saved to its specified file path.
        """
        if self.hparams.save_outputs is not None:
            for stage in ['train', 'val', 'test']:
                if stage in self.output_storage:
                    settings = self.hparams.save_outputs[stage]
                    df = pd.DataFrame(self.output_storage[stage])
                    df.to_csv(settings.file)

    def metric_data_from_batch(self, batch, keys):
        """Extract additional data required by a metric from the batch.

        :param batch: The current batch dict.
        :param keys: List of keys to extract.
        :return: List of values corresponding to the given keys.
        """
        return [batch[k] for k in keys]

    def add_param_to_logger(self, name, param):
        """Register an additional parameter to be logged at each training step.

        :param name: Display name for logging.
        :param param: Parameter tensor to log.
        """
        self._params_to_log.append((name, param))

    def log_params(self):
        """Log all registered parameter values at the current training step."""
        for name, param in self._params_to_log:
            self.log(f"{name}", param, on_step=True)

    def clean_checkpoint(self, checkpoint):
        """Remove specified keys from the checkpoint dict before saving.

        Keys to remove are defined in the class attribute ``remove_from_checkpoint``
        as dot-separated paths (e.g. ``"hyper_parameters.metrics"``).
        """
        if hasattr(self, "remove_from_checkpoint"):
            for prop in self.remove_from_checkpoint:
                path = prop.split(".")
                obj = checkpoint
                for i, name in enumerate(path):
                    if i == len(path) - 1 and name in obj:
                        del obj[name]
                        break

                    if name in obj:
                        obj = obj[name]
                    else:
                        break


if __name__ == "__main__":
    pass
