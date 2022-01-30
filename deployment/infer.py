"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""

import argparse
import os
from typing import List

import cv2
import numpy as np
import onnxruntime as rt
from loguru import logger

from utils import calculate_coord_length


class Model:
    """Class to load trained model and perform inferencing"""

    def __init__(self, model_path: str = "./models/model.onnx") -> None:
        """Constructor to load model

        Args:
            model_path (str, optional): model path. Defaults to "./models/model.onnx".
        """
        # load model

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads onnx model"""
        try:
            self.model = rt.InferenceSession(self.model_path)
        except Exception as e:
            logger.error(f"Not able to load model :{e}")

    def _preprocess(self, image: np.ndarray, img_dim: List[int] = [320, 320]):
        """Preprocessing function to do transformations required before inferecing from model

        Args:
            image (np.ndarray): image to infer on
            img_dim (List[int], optional): image size dimentions. Defaults to [320, 320].

        Returns:
            image: transformed image to infer from model
            image_viz: ndarray for visualisation purpose
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # logger.debug(image.shape)
        if len(image.shape) == 2:
            image = np.dstack([image, image, image])
        image_viz = image.copy()
        image = cv2.resize(image, (img_dim[0], img_dim[1]))

        image = image.astype("float32") / 255
        image -= mean
        image /= std
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image, image_viz

    def _postprocess(self, pred_mask: np.ndarray, image_viz: np.ndarray) -> np.ndarray:
        """Function to postprocess the output mask

        Args:
            pred_mask (np.ndarray): predicted mask from model
            image_viz (np.ndarray): original image

        Returns:
            mask[np.ndarray]: post processed mask
        """
        pred_mask = pred_mask.squeeze()
        pred_mask = cv2.resize(pred_mask, (image_viz.shape[1], image_viz.shape[0]))
        mask = pred_mask > 0.5
        mask = mask.astype(np.uint8) * 255
        return mask

    def predict(self, image: np.ndarray):
        """Fucntion to infer from model

        Args:
            image (np.ndarray): input image

        Returns:
            mask: predicted mask from model
            overlay_image: overlay image for visulaisation

        """
        try:
            # preprocess
            processed_img, image_viz = self._preprocess(image.copy())
            # infer from model
            pred_mask = self.model.run(None, {"input": processed_img})[0]

            # post process mask to original size
            mask = self._postprocess(pred_mask, image_viz)

            overlay_image = cv2.addWeighted(
                image_viz, 0.5, np.dstack([mask, mask, mask]), 0.5, 1, image_viz.copy()
            )
            return mask, overlay_image
        except Exception as e:
            logger.error(e)
            return None, None, None


"""Command

python deployment/infer.py \
    --model_path ./deployment/models/model.onnx \
    --output_dir  ./prediction_output/ \
    --image_path ./data/training_data/test/midaxial2056.png
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer from model")
    parser.add_argument(
        "-mp",
        "--model_path",
        help="onnx model path",
        default="./deployment/models/model.onnx",
    )
    parser.add_argument(
        "-od", "--output_dir", help="output dir", default="./prediction_output/"
    )
    parser.add_argument("-ip", "--image_path", help="image path", required=True)
    args = parser.parse_args()

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    try:
        if not (os.path.isfile(args.model_path) or os.path.isfile(args.image_path)):
            raise Exception("Wrong model or image path.")

        # load model
        predictor = Model(args.model_path)
        # load image
        image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)

        logger.info("Predicting from model")
        mask, overlay_image = predictor.predict(image)

        logger.info("Saving output")
        cv2.imwrite(
            os.path.join(output_path, f"output_{os.path.basename(args.image_path)}"),
            overlay_image,
        )
    except Exception as e:
        logger.error(f"Exception raised : {e}")