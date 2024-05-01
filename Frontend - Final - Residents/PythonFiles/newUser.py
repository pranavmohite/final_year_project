import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump
import os
import csv
import shutil
import streamlit as st

def vgg_face():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

model = vgg_face()
model.load_weights('PythonFiles/vgg_face_weights.h5')


def addUser(csv_file,data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        # file.write("\n")

def count_users(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count

def count_folders(directory):
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return len(folders)

def count_images_per_class(directory):
    num_images_per_class = []
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            num_images_per_class.append(num_images)
    return num_images_per_class

def TrainNewUsers():
    train_dir='PythonFiles/newUser'
    Train_Data=tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        zoom_range = {0.5,1},
        rotation_range = 45,
        fill_mode = 'nearest',
        rescale=1/255.0
    ).flow_from_directory(train_dir,batch_size=16,subset="training",target_size=(224,224),shuffle=False)


    print(list(Train_Data.class_indices.keys()))

    new_user_embedding_vector = model.predict(Train_Data, steps=len(Train_Data), verbose=1)
    
    
    old_embedding_vector = np.load('PythonFiles/embedding_vector_updated.npy')
    y_train = np.load('PythonFiles/embedding_labels_updated.npy')

    newUser_path = 'PythonFiles/newUser'
    Dataset_path = "PythonFiles/dataset.csv"

    new_user_labels = count_folders(newUser_path)  
    dataset_size = count_users(Dataset_path)-1
    

    num_images_per_class_new = count_images_per_class(newUser_path)

    labels_old = np.load('PythonFiles/embedding_labels_updated.npy')
    print(labels_old)

    labels_new = np.repeat(np.arange(dataset_size, dataset_size + new_user_labels), num_images_per_class_new)
    print(labels_new)
    y_train = np.concatenate((labels_old, labels_new))
    print(y_train)

    embedding_vector = np.append(old_embedding_vector, new_user_embedding_vector, axis=0)

    np.save('PythonFiles/embedding_vector_updated',embedding_vector)
    np.save('PythonFiles/embedding_labels_updated',y_train)

    X_train,X_test,y_train,y_test=train_test_split(embedding_vector,y_train,test_size=0.1,shuffle=True, stratify=y_train,random_state=42)

    X_train,y_train = shuffle(X_train,y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=128)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = SVC(kernel='linear',C=2.,class_weight='balanced',decision_function_shape='ovo',probability=True)
    clf.fit(X_train, y_train)

    dump(scaler, 'PythonFiles/scaler_updated.joblib') 
    dump(pca, 'PythonFiles/pca_model_updated.joblib')
    dump(clf, 'PythonFiles/SVC_updated.joblib') 

    newUserData = list(Train_Data.class_indices.keys())
    counter = dataset_size
    for person in newUserData:
        data = (person,counter)
        addUser(csv_file = Dataset_path,data = data)
        counter = counter +1
        shutil.rmtree(f"PythonFiles/newUser/{person}")


if __name__ =="__main__":
    TrainNewUsers()
