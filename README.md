# Churn predicition - Sparkify



## Table of Contents

1. [Project Motivation](#motivation)
2. [Installation and Libraries](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

This project was made as final capstone on Udacity Data Science Nanodegree by Pedro Rocha.

Sparkify is a simulated digital music service as Spotify or Pandora.

Millions of users stream music from the app every day using the free or paid version.

They can upgrade, downgrade or cancel their account at any time. So it's important to make sure that users love the service!

The full dataset is 12GB but in this notebook we'll analyze and deploy ML models in a small subset.

The objective of the project is create a model that can predict users churn.

It's important to predict when a users are going to cancell their accounts in order to act, offering discounts and incentives.

First we'll clean the data set, analyze the data and then perform feature engineering to deploy a ML classificator to achieve our goal.

For this project we'll use **SPARK**, so after analyze this small subset we can scale the analysis to the full dataset in a spark cluster.


## Installation and Libraries  <a name="installation"></a>

Python version: `3.8.5`

You will need to install SPARK and the following libraries:

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, desc, lit, min, max, udf, when, count
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier,GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import  StandardScaler,VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## File Descriptions <a name="files"></a>

Repository structure:
    
    churn_prediction_spark/                  # folder
    ├── README.md                            # read.me file 
    ├── mini_sparkify_event_data.rar         # Data set - subset
    ├── Sparkify.ipynb                       # Project Notebook. Analysis and ML model


## Results<a name="results"></a>

We have created three machine learning classificators to predict churn with good results:

* **Logistic Regression:** F1-Score = 0.662745


* **Random Forest:** F1-Score = 0.71197


* **GBT Classifier:** F1-Score = 0.68674

It was a very interesting project and I hope you enjoyed it as much as I did.

Learn spark for this project was really motivating, I'm happy to increase my skills and be able to handle such a powerful tool.

The next challenge it's to deploy the code in a spark cluster.

I think that the model could perform better with the full data set working to perform the predictions.

Also I think it could be interesting to work with the entire data set in order to create some features that take into account time. For example: activity in the last month or account age

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

All the data was provided by Starbucks.

Code and analysis by Pedro Rocha.

Any feedback is welcome!