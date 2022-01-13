import pandas as pd
import tensorflow
import keras
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("ShortSurvey.csv", sep=";")

print(data.head())

data = data [["In reference to the past term, how would you rate your mental health?" ]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

