"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 18th January 2022
Last-modified Monday, 18th January 2022
"""

import argparse
import os
from typing import Union

import torch
from loguru import logger
from omegaconf import OmegaConf

from loader import load_model


def convert(log_dir: Union[str, os.PathLike], model_output_dir: Union[str, os.PathLike]):
    """This function converts smp model to onnx

    Args:
        log_dir (Union[str, os.PathLike]): log directory path
        model_output_dir (Union[str, os.PathLike]): output dir to save model
    """
    try:
        # load configurations
        cfg = OmegaConf.load(os.path.join(log_dir, ".hydra/config.yaml"))

        model = load_model(log_dir, cfg)

        logger.info("Model loaded")

        dummy_input = torch.randn(1, 3, 320, 320, device="cpu")
        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(model_output_dir, "model.onnx"),
            input_names=["input"],
            output_names=["output"],
            verbose=False,
            opset_version=11,
        )
        logger.info(f"Model saved at :{os.path.join(model_output_dir, 'model.onnx')}")
    except Exception as e:
        logger.error(f"Exception raised while converting to onnx: {e}")


"""Command

python model_utils/onnx_conversion.py \
    --log_dir  ./outputs/2022-01-17/03-50-40 \
    --output_dir ./deployment/models/ 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model on test-data")
    parser.add_argument("-ld", "--log_dir", help="log dir of hydra")
    parser.add_argument("-od", "--output_dir", help="model output dir", default="../delpoyment/models/")
    args = parser.parse_args()

    model_output_dir = args.output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    log_dir = args.log_dir
    # test model
    convert(log_dir, model_output_dir)
