#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import sys
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
get_ipython().run_line_magic('matplotlib', 'inline')

def gaussian(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def load_img(img_path):
    img = mpimg.imread(img_path).astype('float64')/255.0
    R_Channel = img[:,:,0]
    G_Channel = img[:,:,1]
    B_Channel = img[:,:,2]
    luminence = (0.2989*R_Channel) + (0.5870*G_Channel) + (0.1140*B_Channel)
    luminence = convolve(luminence, gaussian(size=5, sigma=1))
    return luminence

def get_img_grad(img):
    sobel_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]], np.float64)
    sobel_y = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]], np.float64)
    Fx = ndimage.filters.convolve(img, sobel_x)
    Fy = ndimage.filters.convolve(img, sobel_y)
    grad_magnitude = np.hypot(Fx, Fy)
    grad_magnitude = grad_magnitude/grad_magnitude.max() * 1.0
    grad_direction = np.arctan2(Fy,Fx)
    grad = (grad_magnitude, grad_direction)
    return grad

def non_maximal_suppression(grad):
    grad_magnitude, grad_direction = grad
    pi = np.pi
    A, B = grad_magnitude.shape
    thinned_img = np.zeros_like(grad_magnitude, np.float64)
    grad_direction[grad_direction < 0] += pi
    for i in range(1, A-1):
        for j in range(1, B-1):
            x = 1.0
            y = 1.0
            if (0 <= grad_direction[i][j] < pi/8) or (7*pi/8 <= grad_direction[i][j] <= pi):
                x = grad_magnitude[i][j+1]
                y = grad_magnitude[i][j-1]
            elif (pi/8 <= grad_direction[i][j] < 3*pi/8):
                x = grad_magnitude[i+1][j-1]
                y = grad_magnitude[i-1][j+1]
            elif (3*pi/8 <= grad_direction[i][j] < 5*pi/8):
                x = grad_magnitude[i+1][j]
                y = grad_magnitude[i-1][j]
            elif (5*pi/8 <= grad_direction[i][j] < 7*pi/8):
                x = grad_magnitude[i-1][j-1]
                y = grad_magnitude[i+1][j+1]
            if(grad_magnitude[i][j] >= x) and (grad_magnitude[i][j] >= y):
                thinned_img[i][j] = grad_magnitude[i][j]
            else:
                thinned_img[i][j] = 0.0  
    return thinned_img

def thresholding_edges(img, T_low=0.10, T_high=0.25):
    T_high = T_high*1
    T_low = T_low*1
    weak_edge = np.zeros_like(img, dtype=np.float64)
    strong_edge = np.zeros_like(img, dtype=np.float64)
    def_not_edge = np.zeros_like(img, dtype=np.float64)
    threshold_img = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] >= T_high):
                strong_edge[i][j] = 1.0
                threshold_img[i][j] = 1.0
            elif(img[i][j] < T_high) and (img[i][j] >= T_low):
                weak_edge[i][j] = .5
                threshold_img[i][j] = .5
            else:
                def_not_edge[i][j] = 1
    return (def_not_edge, weak_edge, strong_edge, threshold_img)

def hysteresis_util(i, j, v, img):
    A, B = img.shape
    if(i<0) or (j<0) or (i>=A) or (j>=B) or (v[i][j] == True):
        return v, img
    v[i][j] = True
    if(img[i][j] == 0):
        return v, img
    img[i][j] = 1.0
    RowN = [-1, -1, -1,  0, 0,  1, 1, 1]
    ColN = [-1,  0,  1, -1, 1, -1, 0, 1]
    for k in range(8): 
        v, img = hysteresis_util(i + RowN[k], j + ColN[k], v, img)

    return v, img

def hysteresis(img):
    A, B = img.shape
    v = np.zeros_like(img, dtype=np.bool)
    for i in range(A):
        for j in range(B):
            if(v[i][j] == False) and (img[i][j] == 1):
                v, img = hysteresis_util(i, j, v, img)
    for i in range(A):
        for j in range(B):
            if(img[i][j] != 1):
                img[i][j] = 0
    
    return img

ROOT_DIR = os.path.dirname(os.getcwd())
print('Root Directory : ', ROOT_DIR, '\n')
IMAGES_DIR = os.path.join('/', ROOT_DIR, 'Data')
print('Images Directory : ', IMAGES_DIR, '\n')

T_low = 0.05
T_high = 0.25

images = os.listdir(IMAGES_DIR)

for image in images:
    img_path = os.path.join('/', IMAGES_DIR, image)
    img = load_img(img_path)

    plt.figure()
    plt.title('Luminence Image')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    
    grad = get_img_grad(img)
    (grad_magnitude, grad_direction) = grad
    
    plt.figure()
    plt.title('Gradient Magnitude')
    plt.imshow(grad_magnitude, cmap='gray')

    plt.figure()
    plt.title('Gradient Direction')
    plt.imshow(grad_direction, cmap='gray')
    
    thinned_img = non_maximal_suppression(grad)

    plt.figure()
    plt.title('Thinned Edge Image')
    plt.imshow(thinned_img, cmap='gray')
    
    edge_type = thresholding_edges(thinned_img, 0.07, 0.25)
    def_not_edge, weak_edge, strong_edge, threshold_img = edge_type

    plt.figure()
    plt.title('Definitely Not Edge Image')
    plt.imshow(def_not_edge, cmap='gray')

    plt.figure()
    plt.title('Weak Edge Image')
    plt.imshow(weak_edge, cmap='gray')

    plt.figure()
    plt.title('Strong Edge Image')
    plt.imshow(strong_edge, cmap='gray')

    plt.figure()
    plt.title('Threshold Image')
    plt.imshow(threshold_img, cmap='gray')

    final_canny_image = hysteresis(threshold_img)

    plt.figure()
    plt.title('Canny Image')
    plt.imshow(final_canny_image, cmap='gray')
    
    
    print('Canny Edge Detection done for ' + image, '\n')


# In[ ]:




