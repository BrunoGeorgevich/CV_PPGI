#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:58:47 2019

@author: bruno
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from time import time

def imshow(*imgs):    
    for i,img in enumerate(imgs):
        cv2.namedWindow(f"{i}", cv2.WINDOW_KEEPRATIO)
        cv2.imshow(f"{i}", img)
    
    while cv2.waitKey(1) != ord('q'):
        pass
    
    cv2.destroyAllWindows()
    
def plot_images(imgs, rows, cols, titles=[], initial_idx = 1, status=None):        
    title_idx = 0
    for i, img in enumerate(imgs, start=initial_idx):
        plt.subplot(rows,cols,i)
        if len(titles) > title_idx:
            plt.title(titles[title_idx])
            title_idx += 1
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
    if status is not None and status == 'show':
        plt.show()
    
def apply_filter(n, W, img):  
    img = img.copy()
    if len(img.shape) == 3:
        h,w,c = img.shape
    elif len(img.shape) == 2:
        h,w = img.shape
        c = 1
    
    assert(n % 2 == 1)
    assert(W.shape[0] == n and W.shape[1] == n and len(W.shape) == 2)

    if c > 1:
        for idx, channel in enumerate(cv2.split(img)):
            img[:,:,idx] = convolve(channel,W)
    else:
        img = convolve(img, W)
    
    return img

def filter_images(n, W, imgs, filtered_imgs):
    for img in imgs:
        filtered_imgs.append(apply_filter(n,W,img))
    

imgs = [cv2.imread(f"{i}.jpg").astype(np.float32)/255 for i in range(1,4)]

lpf_3  = np.array([1/(3**2) for i in range(3*3)], dtype=np.float32).reshape((3,3))
lpf_5  = np.array([1/(5**2) for i in range(5*5)], dtype=np.float32).reshape((5,5))
lpf_7  = np.array([1/(7**2) for i in range(7*7)], dtype=np.float32).reshape((7,7))
lpf_11 = np.array([1/(11**2) for i in range(11*11)], dtype=np.float32).reshape((11,11))

filtered_imgs = []
filter_images(3, lpf_3, imgs, filtered_imgs)    
filter_images(7, lpf_7, imgs, filtered_imgs)
filter_images(11, lpf_11, imgs, filtered_imgs)    

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-baixa para diferentes kernels")

plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs, 4, 3, ['','Kernel 3x3','','','Kernel 7x7','','','Kernel 11x11','',], 4, 'show')

plt.savefig("LPF.png")
plt.close('all')


#%%

hpf_3 = np.array([[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]], dtype=np.float32)
    
hpf_5 = np.array([[ 0,-1,-1,-1, 0],
                  [-1,-2,-4,-2,-1],
                  [-1,-4,36,-4,-1],
                  [-1,-2,-4,-2,-1],
                  [ 0,-1,-1,-1, 0]], dtype=np.float32)
    
hpf_7 = np.array([[ 0, 0, 0,-1, 0, 0, 0],
                  [ 0,-1,-2,-4,-2,-1, 0],
                  [ 0,-2,-4,-8,-4,-2, 0],
                  [-1,-4,-8,80,-8,-4,-1],
                  [ 0,-2,-4,-8, 4,-2, 0],
                  [ 0,-1,-2,-4,-2,-1, 0],
                  [ 0, 0, 0,-1, 0, 0, 0]], dtype=np.float32)
    
filtered_imgs = []
filter_images(3, hpf_3, imgs, filtered_imgs)   
filter_images(5, hpf_5, imgs, filtered_imgs)   
filter_images(7, hpf_7, imgs, filtered_imgs)   

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-alta para diferentes kernels")

plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs, 4, 3, ['','Kernel 3x3','','','Kernel 5x5','','','Kernel 7x7','',], 4, 'show')

plt.savefig("HPF.png")
plt.close('all')
#%%

line_3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    
line_5 = np.array([[-2, -1, 0, 1, 2],
                   [-2, -1, 0, 1, 2],
                   [-4, -2, 0, 2, 4],
                   [-2, -1, 0, 1, 2],
                   [-2, -1, 0, 1, 2]], dtype=np.float32)
    
line_7 = np.array([[-3,-2,-1, 0, 1, 2, 3],
                   [-3,-2,-1, 0, 1, 2, 3],
                   [-3,-2,-1, 0, 1, 2, 3],
                   [-6,-4,-2, 0, 2, 4, 6],
                   [-3,-2,-1, 0, 1, 2, 3],
                   [-3,-2,-1, 0, 1, 2, 3],
                   [-3,-2,-1, 0, 1, 2, 3]], dtype=np.float32)
    
gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]    
    
filtered_imgs = []
filter_images(3, line_3, gray_imgs, filtered_imgs)   
filter_images(5, line_5, gray_imgs, filtered_imgs)   
filter_images(7, line_7, gray_imgs, filtered_imgs)   

plt.figure(1, (16,9))
plt.suptitle("Filtro que detecta arestas para diferentes kernels")

plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs, 4, 3, ['','Kernel 3x3','','','Kernel 5x5','','','Kernel 7x7','',], 4, 'show')

plt.savefig("Line.png")
plt.close('all')
#%%

def hybrid_image(im1, im2, W1, W2, lf, hf, n):
    im1 = im1.copy()
    im2 = im2.copy()
    
    assert(n % 2 == 1)
    assert(W1.shape[0] == n and W1.shape[1] == n and len(W1.shape) == 2)
    assert(W2.shape[0] == n and W2.shape[1] == n and len(W2.shape) == 2)
    
    for idx, channel in enumerate(cv2.split(im1)):
        im1[:,:,idx] = convolve(channel,W1)
        
    im2 = convolve(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),W2)
    im2 = cv2.merge((im2,im2,im2))

    return cv2.addWeighted(im1, lf, im2, hf, 0)

imgs = [cv2.imread(f"{i}.jpg").astype(np.float32)/255 for i in range(1,4)]
fg = cv2.imread('4.jpg').astype(np.float32)/255

hybrid_imgs = []
hybrid_imgs += [hybrid_image(imgs[i], fg, lpf_3, line_3, 0.8, 0.2, 3) for i in range(len(imgs))]
hybrid_imgs += [hybrid_image(imgs[i], fg, lpf_5, line_5, 0.8, 0.2, 5) for i in range(len(imgs))]
hybrid_imgs += [hybrid_image(imgs[i], fg, lpf_7, line_7, 0.8, 0.2, 7) for i in range(len(imgs))]


plt.figure(1, (16,9))
plt.suptitle("Imagens híbridas para diferentes kernels")

plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(hybrid_imgs, 4, 3, ['','Kernel 3x3','','','Kernel 5x5','','','Kernel 7x7','',], 4, 'show')

plt.savefig("Hybrid.png")
plt.close('all')


#%%