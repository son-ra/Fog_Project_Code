#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

links_and_labels_loop = pd.read_csv('/home/smmrrr/Fog_Imaging_Project/Fog_Project_Code/unlabeled_photos_for_model.csv')
image_dir = '/home/smmrrr/surfline/'

# Load images and labels
data = []
labels = []

for index, row in links_and_labels_loop.iterrows():
    img_path = os.path.join(image_dir, row['photo'])
    img = load_img(img_path, target_size=(100, 100))  # Specify the target size of your images
    img_array = img_to_array(img)

    # Edge detection using Canny algorithm
    gray_image = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Stack the original image and edges as input
    img_with_edges = np.dstack([img_array, edges])
    
    data.append(img_with_edges)
    labels.append(row['photo'])

with open('edge_detection_unlabeled_photos.pkl', 'wb') as file:
    pickle.dump(data, file)


with open('edge_detection_unlabeled_photos_name_index.pkl', 'wb') as file:
    pickle.dump(labels, file)
