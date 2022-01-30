"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""


import json
import io
from typing import List

import numpy as np
from PIL import Image
from loguru import logger
import cv2
import math


def interpolation(d: List, x: List):
    """Interpolates line points

    Args:
        d (List): List of 2 points [(row,col),(row,col)]
        x (List): List of [row]

    Returns:
        int]: Interpolated column value
    """
    output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1]) / (d[1][0] - d[0][0]))
    return output


class NpEncoder(json.JSONEncoder):
    """Transformation class for json response

    Args:
        json ([type]): Dictionary
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def read_imagefile(file) -> Image.Image:
    """function to load image data from post request body

    Args:
        file ([type]): file to load image data

    Returns:
        Image.Image: np.ndarray image to process from model
    """
    image = np.array(Image.open(io.BytesIO(file)))
    logger.debug(image.shape)
    return image
