"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""


import json
import time

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile

from infer import Model
from utils import NpEncoder, read_imagefile

app = FastAPI(title="SpineX App")

# load the model
predictor = Model("./models/model.onnx")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.2f} seconds"
    return response


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    org_image = read_imagefile(await file.read())
    mask, overlay_image = predictor.predict(org_image)
    if mask is None:
        msg = {"status": "100"}
    else:

        msg = {"status": "200"}
    # logger.debug(msg)
    return json.dumps(msg, cls=NpEncoder)
