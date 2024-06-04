# PCA-CNN
A Principal Component Analysis-Convolutional Neural Network (PCA-CNN)-based method for optimizing equipment operating parameters. It is used to obtain the optimal operating parameters and the optimal production mode that satisfy a multi-objective function.

## Environment Details
```
python==3.8.8
numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
tensorflow==2.8
```

## Overview
The transportation distance of coal flow in mines is long, and the environment is complex. The transportation equipment usually adopts a constant speed mode, resulting in much energy waste. To solve the problems of high energy consumption and severe waste of coal flow transportation system equipment, a device operation parameter optimization method based on principal component analysis convolutional neural network (PCA-CNN) is proposed based on analyzing the characteristics of coal flow transportation system. A multi-objective function has been established with the transportation time, transportation cost, and equipment utilization rate of coal mine belt conveyors and other equipment as optimization objectives, and the operation variables of transportation equipment, such as transportation speed, transportation distance, and start-up time, as decision variables. The principal component analysis method is used to determine the weights of each objective function, and a convolutional neural network is used to iteratively train the actual production data samples of coal mines under multiple constraint conditions to obtain the optimal operating parameters and production mode of coal flow transportation equipment that meet the multi-objective function. Engineering feasibility simulations were conducted through plant simulation to compare the operational parameters of coal flow transportation equipment before and after optimization. It was found that the objective functions of each experiment were optimized to varying degrees, with optimization levels ranging from 17% to 25%.

## File Description
```
*Data (folder): Contains test and training sets
*Log (folder): Save the training model and parameters
*Image (folder): Store training and prediction result images
*Input_data.py: Responsible for reading data and generating batches
*Model. py: Responsible for implementing our neural network model
*Training.py: Responsible for implementing model training and evaluation
*Test. py: Extract a piece of data from the test set for prediction.
```
