import os
from typing import List, Optional

import hydra
import pytorch_ie as pie
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # TODO: implement pie.data.DatasetDict
    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset: pie.data.DatasetDict = hydra.utils.instantiate(config.dataset)

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{config.taskmodule._target_}>")
    taskmodule: pie.taskmodules.taskmodule.TaskModule = hydra.utils.instantiate(
        config.taskmodule, dataset=dataset
    )

    # TODO: add parameter "train_split" to config
    log.info(f"Prepare taskmodule with train data split: {config.train_split}")
    taskmodule.prepare(dataset[config.train_split])

    # TODO: implement pie.datamodule.DataModule
    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pie.datamodule.DataModule = hydra.utils.instantiate(
        config.datamodule, dataset=dataset, taskmodule=taskmodule
    )

    # TODO: how to pass parameters from taskmodule to the model?
    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pie.core.pytorch_ie.PyTorchIEModel = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        taskmodule=taskmodule,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # TODO: add parameter "save_dir" to config (with "." as default?)
    save_dir = config["save_dir"]
    if not os.path.isabs(save_dir):
        config.save_dir = os.path.join(hydra.utils.get_original_cwd(), save_dir)
    # TODO: add parameter "push_to_hub" to config
    log.info(f"Save taskmodule to {save_dir} [push_to_hub={config.push_to_hub}]")
    taskmodule.save_pretrained(save_directory=save_dir, push_to_hub=config.push_to_hub)

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        taskmodule=taskmodule,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
        model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        log.info(f"Save best model to {save_dir} [push_to_hub={config.push_to_hub}]")
        model.save_pretrained(save_directory=save_dir, push_to_hub=config.push_to_hub)

    # Return metric score for hyperparameter optimization
    return score