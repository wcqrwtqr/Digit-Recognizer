#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:41:15 2018

@author: mohammedalbatati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import the accuracy lib
from sklearn.metrics import accuracy_score
model_score = pd.DataFrame()
# Array for the storing the accuracy / model type data
def saving_model_scoring(model, X_test, y_test, classifier_name):
    y_pred_ = model.predict(X_test)
    acc_score_ = accuracy_score(y_test, y_pred_)
    name = str(classifier_name)
    model_name = [name]
    model_score_df = pd.DataFrame({'Accuracy':acc_score_ }, index =model_name )
    return model_score_df
    

#===================== Importing the data======================
train_dataset = pd.read_csv('train.csv')
imges_to_Digitize = pd.read_csv('test.csv')

#Uncomment to check if the train data are distributed evenly
#train_dataset['label'].value_counts(normalize=True)

## to get the image for showing it to console
#img = test_dataset.iloc[3,:].values.reshape(28,28)
#plt.imshow(img)
#plt.show()
#===================== Data Preparation======================
X = train_dataset.drop(['label'], axis=1).copy()
y = train_dataset['label'].copy()

# Clean the data set for memory saving
#del train_dataset

#scalling the X values to ease the modeling computation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scalled = sc.fit_transform(X)
images_to_Digitize_scalled = sc.fit_transform(imges_to_Digitize)


## to get the image for showing it to console
#img = X_scalled[1].reshape(28,28)
#plt.imshow(img)
#plt.show()

# Splitting the data for the training of the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scalled, y,
                                                    test_size=0.2,
                                                    random_state=0)
# Clean the data set for memory saving
del X , y
del X_scalled
del imges_to_Digitize
#===================== Modelling======================
########## Using the random forest model #################
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=200, verbose=2)
random_forest_model.fit(X_train, y_train)

# updating the score data frame
x_score = saving_model_scoring(model=random_forest_model,
                     X_test=X_test, 
                     y_test=y_test, 
                     classifier_name='RandomForst')
model_score = model_score.append(x_score)


########## Using the GaussianNB model #################
from sklearn.naive_bayes import GaussianNB
naiv_model = GaussianNB()
naiv_model.fit(X_train, y_train)

x_score = saving_model_scoring(model=naiv_model,
                     X_test=X_test, 
                     y_test=y_test, 
                     classifier_name='naive_bayes')
model_score = model_score.append(x_score)


########## Using the Decision tree model #################
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

x_score = saving_model_scoring(model=tree_model,
                     X_test=X_test, 
                     y_test=y_test, 
                     classifier_name='Decision tree')
model_score = model_score.append(x_score)


########## Using the ANN model #################
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.activations import relu , softmax

ann_model = Sequential()
ann_model.add(Dense(200, activation=relu , input_dim=784))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(10, activation=softmax))
ann_model.summary()

# reshaping the y_test & y_train for the ANN model to avoid run errors
y_train_shaped = y_train.values.reshape((-1, 1))
y_test_shaped = y_test.values.reshape((-1, 1))

# compiling the model using Adam optimizer
ann_model.compile(Adam(), 'sparse_categorical_crossentropy',metrics=['accuracy'])

#Training the model
history = ann_model.fit(X_train , y_train_shaped , batch_size=2000, epochs=30, verbose=1, 
                        validation_data=(X_test , y_test_shaped))

# get the accuracy score and updating our model_score array
y_pred = ann_model.predict(X_test)
y_ann_pred = []
for i in y_pred:
    x = i.argmax()
    y_ann_pred.append(x)
#y_ann_pred = y_ann_pred
xx =accuracy_score(y_test_shaped , y_ann_pred)
xy = ['ANN']
new_df = pd.DataFrame({'Accuracy': xx} , index = xy)
model_score = model_score.append(new_df)
del x,y_ann_pred, xx,xy


########## Using the CNN model #################
from keras.layers import  MaxPool2D, Flatten, Convolution2D
cnn_model = Sequential()
cnn_model.add(Convolution2D(28,3,3, input_shape = (28, 28, 1), activation=relu))
cnn_model.add(MaxPool2D())
cnn_model.add(Flatten())
cnn_model.add(Dense(200, activation=relu))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(200, activation=relu))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(200, activation=relu))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(10, activation=softmax))
cnn_model.summary()

cnn_model.compile(Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Convert the images from csv file to images with proper channels
# X_train
X_train_img = np.ndarray(shape=(len(X_train), 28, 28,1),dtype=np.float32)
for i in range(len(X_train)):
    imged = X_train[i, :].reshape(28,28,1)
    X_train_img[i] = imged

#X_test
X_test_img = np.ndarray(shape=(len(X_test), 28, 28,1),dtype=np.float32)
for i in range(len(X_test)):
    imged = X_test[i , :].reshape(28,28,1)
    X_test_img[i] = imged

history = cnn_model.fit(X_train_img, y_train_shaped,batch_size=1000, 
                        verbose=1, 
                        epochs= 20,
                        validation_data=(X_test_img,y_test_shaped))


#===============================================================
#===================== END of Modelling=========================
# Found the ANN model is having the best accuracy score

#Use the ANN model to generate the prediction of values
y_result = ann_model.predict(images_to_Digitize_scalled)
y_ann_result = []
for i in y_result:
    x = i.argmax()
    y_ann_result.append(x)


# Visualize the images vs the predicted y for a given range
for i in range(2000,2010):
    img = images_to_Digitize_scalled[i, :].reshape(28,28)
    plt.imshow(img)
    plt.show()
    print(y_ann_result[i])

# Writing the results to csv
final_df = pd.DataFrame(y_ann_result, columns= ['Label'])
final_df.to_csv('submission.csv', index_label = 'ImageId')


#===============================================================
#===================== END of Modelling=========================
# Found the CNN model is having the best accuracy score

#Convert the images from csv file to images with proper channels
# i
images_to_Digitize_img = np.ndarray(shape=(len(images_to_Digitize_scalled), 28, 28,1),dtype=np.float32)
for i in range(len(images_to_Digitize_scalled)):
    imged = images_to_Digitize_scalled[i, :].reshape(28,28,1)
    images_to_Digitize_img[i] = imged



#Use the CNN model to generate the prediction of values
y_result_cnn = cnn_model.predict_proba(images_to_Digitize_img)
y_cnn_result = []
for i in y_result_cnn:
    x = i.argmax()
    y_cnn_result.append(x)


# Visualize the images vs the predicted y for a given range
for i in range(9000,9010):
    img = images_to_Digitize_scalled[i, :].reshape(28,28)
    plt.imshow(img)
    plt.show()
    print(y_cnn_result[i])

# Writing the results to csv
final_df_cnn = pd.DataFrame(y_cnn_result, columns= ['Label'])
final_df_cnn.to_csv('submission_cnn.csv', index_label = 'ImageId')

# add one more note to check the Github working or not






