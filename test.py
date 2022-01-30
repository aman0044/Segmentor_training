"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""

import argparse
import json
import os

import segmentation_models_pytorch as smp
from loguru import logger
from omegaconf import OmegaConf

from data_utils import get_loader
from model_utils import load_model


def test(log_dir: os.PathLike):
    """This function evaluates model on test dataset. It returns IOU and dice loss of model on test-dataset.

    Args:
        log_dir (Union[str, os.PathLike]): log directory path
    """
    try:
        # load configurations
        cfg = OmegaConf.load(os.path.join(log_dir, ".hydra/config.yaml"))

        # load model
        model = load_model(log_dir, cfg)
        logger.info("Model loaded")

        # load dataloader
        test_dataloader = get_loader(cfg, "test")

        loss = eval(f"smp.utils.losses.{cfg.training.LOSS}()")
        metrics = [
            smp.utils.metrics.IoU(
                threshold=0.5,
            )  # ignore_channels=[0, 2]),
        ]

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=model,
            loss=loss,
            metrics=metrics,
            device=cfg.model.DEVICE,
        )
        logger.info("Inferencing")
        logs = test_epoch.run(test_dataloader)
        logger.info(json.dumps(logs, indent=4))
    except Exception as e:
        logger.error(f"Exception raised: {e}")


"""Command

python test.py \
    --log_dir  ./outputs/2022-01-17/03-50-40

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model on test-data")
    parser.add_argument("-ld", "--log_dir", help="log dir of hydra")
    args = parser.parse_args()

    log_dir = args.log_dir
    # test model
    test(log_dir)
