# Segmentor Training  

## Contents

* [Purpose](#purpose)
* [Features](#features)
* [Model Training](#model-training)
    * [Data Creation](#data-creation)
    * [Training Data](#training-data)
    * [Training Data location](#training-data-location)
    * [Training Process](#training-process)
* [Model Evaluation](#model-evaluation)
* [Model conversion to onnx](#model-conversion-to-onnx)
* [Deployment](#deployment)  
    * [Pipeline](#pipeline)  
    * [Deployment of pipeline](#deployment-of-pipeline)  
* [Scope of Improvement](#scope-of-improvement)
* [Resources](#resources)

## Purpose

This repository contaions end-to-enf pipeline for data-processing, training and deployment of segmentation models using [SMP](#https://github.com/qubvel/segmentation_models.pytorch) library.

## Features
1. **End-to-End** pipeline for training and deployment
2. **Hydra** configuratiions for training
3. SMP (https://github.com/qubvel/segmentation_models.pytorch) models with **Catalyst**
4. **FastAPI** 
5. **Dockerized** deployment script

## Model Training

### 1. Data Creation

#### **Image Source**:
Kitti dataset

#### **Labelling strategy**:
We can use **CVAT** tool.

##### Steps involved in label creation:
    1. Load images in CVAT tool and create segmentation mask over spinal coord portion.  
    2. Export data into segmentation format. The exported data has mask into specified RGB color. 
    3. Transform label mask into binary mask. (Use: [./notebook/data_utitlity.ipynb](./notebook/data_utitlity.ipynb)). This is the requirement of training script.

##### **Sample:**  
>> 

### 2. Training Data
We should opt for general data splitting strategy: 

| Train| Validation | Test |  
|:----:|:----------:|:----:|  
| 0.7  | 0.2        | 0.1  |  

### 3. Training Data location

All the data is maintained inside **data/** folder.

    training_data
        ├── test
        ├── testannot
        ├── train
        ├── trainannot
        ├── val
        └── valannot

### 4. Training Process

For training the segmentation model I have used [smp](#https://github.com/qubvel/segmentation_models.pytorch) library. This library has simple interface and supports more than 10 architectures with approx 400 encoders.

The training is facilitated by  [**catalyst**](#https://github.com/catalyst-team/catalyst) and [**hydra**](https://github.com/facebookresearch/hydra). Different augmentations implemented using **Albumentation** library. Check [.notebook/training_data_visualisation.ipynb](#notebook/training_data_visualisation.ipynb) notebook for training data visualisation.

#### Steps to start training
1. Create virtual env by installing the requirements kept in **requirements.txt** file.

    ```python 
    conda create --name <env_name> --file requirements.txt
    ```

    You also need to setup jupyter notebook kernel for the same env.

    ```python

    python -m ipykernel install --user --name <env_name> --display-name <env_name>
    ```


2. Training data is kept inside **data** folder. If you want to train model on custom data then it's recommended to transform your data using [./notebook/data_utitlity.ipynb](./notebook/data_utitlity.ipynb) notebook and keep training data in the same **data** folder.

3. All the training configurations are kept inside different config files.  

        ./configs
        ├── config.yaml
        ├── model
        │   └── default.yaml
        ├── processing
        │   └── default.yaml
        └── training
            └── default.yaml
    
    * model/default.yaml --> stores all the model related configs.  
    * processing/defualt.yaml --> stores all configs relates to data processing.  
    * training/default.yaml --> stores configs related to training process.
> Note: Make changes according to your training resources.

4. Training script is kept in **./train.py** file. After tuning the configurations you can start the training process by simply hitting below command.
```python
python train.py  
```
> Note: As i had used hydra you can provide run time arguments to modify the default configs, as well as provide different conbinations of arguments to start different training loops.

5. Logs of training will be stored in **output/{DATE}/{TIME}/** folder. This folder will keep all the training configs, logs, models and tensorboard event files.

    You can check the tensorboard by hitting below command
    ```python
    tensorboard --logdir ./output/
    ```

## Model Evaluation

Evaluate model on test data by using **./test.py** file. You need to provide log output folder path to run this file. 
```python
python test.py \
    --log_dir  ./outputs/xxxx/xxxx
```

Result:

| Images | dice_loss | iou_score  |
|--------|-----------|------------|
| xxxx |  xxx | xxx |

For result visualisation you can use [./notebook/result_visualisation.ipynb](#./notebook/result_visualisation.ipynb) notebook.

## Model conversion to onnx

To convert model **smp** model to **onnx**. 
```python 
python utils/onnx_conversion.py \
    --log_dir  ./outputs/xxxx/xxxx \
    --output_dir ./delpoyment/models/ 
```

For onnx model results verification use [./notebook/smp_to_onnx.ipynb](#./notebook/smp_to_onnx.ipynb) notebook.


## Infer from onnx model

You can also infer from onnx model and save output image by using below command.
```python
python deployment/infer.py \
    --model_path ./deployment/models/model.onnx \
    --output_dir  ./prediction_output/ \
    --image_path ./data/training_data/test/test.jpg

```

## Deployment

For the deployment, I have used **FastAPI** and **Docker** to create application. All the deployment code is kept inside **./deployment** folder.


### Pipeline
>> 

### Deployment of pipeline

For the deployment of pipeline on docker all you need to do is to move inside deployment folder and run below commands
> Note:  Make sure you have onnx model kept inside **./deployment/models/model.onnx path**  .

1. Build docker image
    ```python
    docker build -t spinex:latest .
    ```
2. Start docker container
    ```python
    docker run -p 8000:8000 --name spinex spinex
    ```
3. You can use FastAPI dashboard to test the API (url: http://0.0.0.0:8000/docs/) OR you can hit CURL request from Postman.  

    **Request**
    ```python
    curl -X 'POST' \
    'http://0.0.0.0:8000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@mtest.png;type=image/png'
    ```
    **Response**
    ```python 
    {'status': '200'}}
    ```

#### Status codes

| Var | Description                                                                         |
| --- | ----------------------------------------------------------------------------------- |
| 200 | Success                                                                             |
| 100 | Problem while predicting from model                                                 |


## Scope of Improvement

1. DVC is not utilised in current version
2. Streamlit is always a good choice for representation of model output, which is indeed missing in current version.

## Resources

#### Scripts
| Details |   Type   |     Link      | 
|:-------|:-------:|:-------------|
| Training Script| python file | [train.py](./train.py)|
| Evaluation Script| python file | [test.py](./test.py)|
| Inferencing Script| python file | [deployment/infer.py](./deployment/infer.py)|
| Onnx conversion Script| python file | [./model_utils/onnx_conversion.py](./model_utils/onnx_conversion.py)|
| Deployment application script | python file | [./deployment/app.py](./deployment/app.py)
|**Data Operations** | | |
| Label Binary Conversion | notebook | [./notebook/data_utitlity.ipynb](./notebook/data_utitlity.ipynb)|
| Data Splitting | notebook | [./notebook/data_utitlity.ipynb](./notebook/data_utitlity.ipynb) |
| Training Data Visualisation | notebook | [./notebook/training_data_visualisation.ipynb](./notebook/training_data_visualisation.ipynb)|
| Onnx Model Verification | notebook | [./notebook/smp_to_onnx.ipynb](./notebook/smp_to_onnx.ipynb)|

















