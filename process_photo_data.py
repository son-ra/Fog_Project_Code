#!/usr/bin/env python
# coding: utf-8

import skimage
import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import re
import numpy as np
import seaborn as sns
## for the stats class
import scipy.signal as sg
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.decomposition import SparsePCA
import pywt
from scipy import stats
import timeit 
#import optshrink as opt # package we create
import numpy as np
# import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
from datetime import datetime
import pytz
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from skimage import io
from patchify import patchify, unpatchify



import collections
# from itertools import chain
# import urllib.request as request
# import pickle 

import numpy as np

import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.pyplot as plt

import skimage.io
import skimage.transform

# import cv2

# from libsvm import svmutil

import Shady.Contrast as sc 




# https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    return (image - local_mean) / (local_var + C)

def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })



links_and_labels_loop = pd.read_csv('/home/smmrrr/Fog_Imaging_Project/Fog_Project_Code/unlabeled_photos_for_model.csv')
image_dir = '/home/smmrrr/surfline/'




fog_aware_stats = pd.DataFrame(columns = ['photo', 'patch' ,'mscn_var','vertical_var','sharpness'
                                          ,'coef_or_var_sharpness','rms_contrast','entropy','dark_channel_prior','color_sat','CF'])
### 3 horizontal patches
x_patch = 1280
y_patch = 240


for f in range(22000, len(links_and_labels_loop)):
    photo = links_and_labels_loop.loc[f, 'photo']
    im = io.imread(image_dir + photo) ## read in file

    patches = patchify(im, (y_patch, x_patch, 3), (y_patch))
    patches_reshape = np.reshape(patches, (patches.shape[0]*patches.shape[1]*patches.shape[2]
                                           , y_patch, x_patch, 3))

    ###loop through each patch
    for i in range(len(patches_reshape)):

        ##get patch and convert it to greyscale and greyscale flattened
        patch = patches_reshape[i, :, :, :]
        patch_grey = rgb2gray(patch)
        flattened_grey = patch_grey.ravel()

        ### variance of mscn coefficients
        mscn = calculate_mscn_coefficients(patch_grey)
        mscn_var = mscn.var()

        ###variance of the vertical product if mscn coefficients (positive, negative mode)
        vertical = mscn[:-1, :] * mscn[1:, :]
        vertical_var = vertical.var()


        ###the sharpness
        ###the coefficient of vaiance of sharpness
        kernel_size=6
        sigma=7/6
        kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(patch_grey, kernel, 'same')
        local_var = local_deviation(patch_grey, local_mean, kernel)
        local_cv = local_var/local_mean
        flattened = patch_grey.ravel()
        sharpness = local_var.mean()
        coef_or_var_sharpness = local_cv.mean()

        ##the contrast energy

        ##root mean square contrast ratio
        rms_contrast = sc.RMSContrastRatio(patch_grey)
        
        ###entropy (H)
        kde_results = gaussian_kde(flattened)

        # # Generate points to evaluate the KDE
        x = np.linspace(np.min(flattened), np.max(flattened), 100)
        H = stats.entropy(kde_results.pdf(x))


        ##the dark channel prior in a pixel-wise
        dark_channel_prior = patch.min()
        dark_channel_prior = dark_channel_prior.astype('int64') #convert from uint8

        ##the color saturation in hsv color space
        saturation = rgb2hsv(patch)[:, :, 1] ##second channel in hue saturation value
        color_sat = saturation.mean()

        ##the colorfulness
        red = patch[:,:,0]
        green = patch[:,:,1]
        blue = patch[:,:,2]

        rg = -1*(green - red) ### it will be squared so order doesnt matter
        yb = .5 * (red + green) - blue
        CF = np.sqrt((rg.std())**2 + (yb.std())**2) + 0.3*np.sqrt((rg.mean())**2 + (yb.mean())**2)

        fog_aware_stats = pd.concat([
            fog_aware_stats,
        pd.DataFrame({
                     'photo':[photo]
                     , 'patch' :[i]
                     ,'mscn_var':[mscn_var ]
                     ,'vertical_var':[vertical_var]
                     ,'sharpness':[sharpness]
                     ,'coef_or_var_sharpness':[coef_or_var_sharpness]
                     ,'rms_contrast':[rms_contrast]
                     ,'entropy':[H]
                     ,'dark_channel_prior':[dark_channel_prior]
                     ,'color_sat':[color_sat]
                     ,'CF':[CF]
                     })
        ])

#         print(i)
#         print(len(fog_aware_stats))
    # if (f % 1000 == 0):
fog_aware_stats.to_csv('fog_aware_stats_horizontal_3_patches_reboot2.csv')



