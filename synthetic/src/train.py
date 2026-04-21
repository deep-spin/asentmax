import os
from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from omegaconf import open_dict
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)

import faulthandler
faulthandler.enable()


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random

    if cfg.get("disable_cuda", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    ckpt_path = cfg.get("ckpt_path", None)

    if hasattr(cfg.callbacks.model_checkpoint, "dirpath") and cfg.callbacks.model_checkpoint.dirpath is not None:
        path_obj = Path(cfg.callbacks.model_checkpoint.dirpath)
        last_ckpt_path = os.path.join(cfg.callbacks.model_checkpoint.dirpath, "last.ckpt")
        log.info(f"====> {last_ckpt_path}")
        if path_obj.exists() and Path(last_ckpt_path).exists():
            log.info(f"Resuming training from the last checkpoint: {last_ckpt_path}")
            ckpt_path = last_ckpt_path
        else:
            path_obj.mkdir(parents=True, exist_ok=True)

    if ckpt_path is not None and cfg.get("skip_data", True):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        with open_dict(cfg):
            log.info(f"Data skip steps: {checkpoint['global_step']}")
            cfg.data.data_skip_steps = checkpoint["global_step"]
        del checkpoint

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model(tokenizer=datamodule.tokenizer)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Extract matmul_precision before instantiation (Trainer doesn't accept it)
    matmul_precision = cfg.trainer.pop('matmul_precision', None)
    if matmul_precision is not None:
        log.info(f"Setting matmul_precision to <{matmul_precision}>")
        torch.set_float32_matmul_precision(matmul_precision)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        torch._dynamo.config.suppress_errors = cfg.get("compile_suppress_errors", True)
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            log.error("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
