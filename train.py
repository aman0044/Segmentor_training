"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""

from loguru import logger
import os

import hydra
import segmentation_models_pytorch as smp
import torch
from catalyst import dl
from catalyst.dl import SupervisedRunner
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import get_loader


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Function to train smp model using configs stored inside ./configs folder

    Args:
        cfg (DictConfig): omegconf configs loaded from configs.yaml file
    """

    # log all the configs
    logger.info(OmegaConf.to_yaml(cfg))

    ### load model
    logger.info("Loading model")
    model = eval(
        f"smp.{cfg.model.ARCH}(encoder_name='{cfg.model.ENCODER}',encoder_weights='{cfg.model.ENCODER_WEIGHTS}',classes={len(cfg.model.CLASSES)},activation=('{cfg.model.ACTIVATION}'))"
    )

    ## load dataloaders
    train_loader = get_loader(cfg, "train")
    valid_loader = get_loader(cfg, "val")

    # training configs
    num_epochs = cfg.training.NUM_EPOCHS
    logdir = cfg.training.LOG_DIR
    loaders = {"train": train_loader, "valid": valid_loader}

    # loss function
    criterion = eval(f"smp.utils.losses.{cfg.training.LOSS}()")
    # optimiser
    optimizer = torch.optim.Adam(
        [
            {"params": model.decoder.parameters(), "lr": cfg.training.DECODE_LR},
            # decrease lr for encoder in order not to permute
            # pre-trained weights with large gradients on training start
            {"params": model.encoder.parameters(), "lr": cfg.training.ENCODER_LR},
        ]
    )
    # lr sechuler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg.training.SCHEDULER_FACTOR,
        patience=cfg.training.SCHEDULER_PATIENCE,
    )
    runner = SupervisedRunner(
        input_key="features", output_key="scores", target_key="targets", loss_key="loss"
    )

    # start training
    logger.info("Training Started")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[
            dl.IOUCallback(input_key="scores", target_key="targets", threshold=0.5),
            dl.DiceCallback(input_key="scores", target_key="targets"),
            dl.EarlyStoppingCallback(
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                patience=5,
                min_delta=0.01,
            ),
            dl.CheckpointCallback(
                logdir=logdir, loader_key="valid", metric_key="loss", minimize=True
            ),
        ],
        logdir=logdir,
        valid_loader="valid",
        valid_metric="loss",
        num_epochs=num_epochs,
        fp16=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
