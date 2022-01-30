from catalyst.utils import unpack_checkpoint, load_checkpoint
import segmentation_models_pytorch as smp
import os
from typing import Dict, Union

__all__ = ["load_model"]


def load_model(log_dir: Union[str, os.PathLike], cfg: Dict):
    """This function load smp model with checkpoints

    Args:
        log_dir (Union[str, os.PathLike]): log directory path
        cfg (Dict): configurations of model

    Returns:
        model: model object
    """
    # load checkpoint
    checkpoint = load_checkpoint(path=os.path.join(log_dir, cfg.training.LOG_DIR, "best_full.pth"))

    # load model
    model = eval(
        f"smp.{cfg.model.ARCH}(encoder_name='{cfg.model.ENCODER}',encoder_weights='{cfg.model.ENCODER_WEIGHTS}',classes={len(cfg.model.CLASSES)},activation=('{cfg.model.ACTIVATION}'))"
    )

    unpack_checkpoint(
        checkpoint=checkpoint,
        model=model,
    )

    return model