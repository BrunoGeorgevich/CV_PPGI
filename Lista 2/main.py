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
        plt.imshow(img, 'gray')
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

def amplitude_and_phase(img):    
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    R = dft_shift[:,:,0]
    I = dft_shift[:,:,1]
    
    A = 20*np.log(cv2.magnitude(R,I))
    A = cv2.normalize(A, None, 0, 1, cv2.NORM_MINMAX)
    P = np.arctan2(I,R)
    P = cv2.normalize(P, None, 0, 1, cv2.NORM_MINMAX)
    return A, P

def filter_images(n, W, imgs, filtered_imgs):
    for img in imgs:
        filtered_img = apply_filter(n,W,img)
        A,P = amplitude_and_phase(filtered_img)
        filtered_imgs['A'].append(A)
        filtered_imgs['P'].append(P)
    

imgs = [cv2.imread(f"{i}.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for i in range(1,4)]

#%%

lpf_3  = np.array([1/(3**2) for i in range(3*3)], dtype=np.float32).reshape((3,3))
lpf_5  = np.array([1/(5**2) for i in range(5*5)], dtype=np.float32).reshape((5,5))
lpf_7  = np.array([1/(7**2) for i in range(7*7)], dtype=np.float32).reshape((7,7))
lpf_11 = np.array([1/(11**2) for i in range(11*11)], dtype=np.float32).reshape((11,11))

filtered_imgs = {'A':[],'P':[]}
filter_images(3, lpf_3, imgs, filtered_imgs)    
filter_images(7, lpf_7, imgs, filtered_imgs)
filter_images(11, lpf_11, imgs, filtered_imgs)    

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-baixa para diferentes kernels - Amplitude")
plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs['A'], 4, 3, ['','Kernel 3x3','','','Kernel 7x7','','','Kernel 11x11','',], 4, 'show')
plt.savefig("LPF_A.png")
plt.close('all')

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-baixa para diferentes kernels - Phase")
plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs['P'], 4, 3, ['','Kernel 3x3','','','Kernel 7x7','','','Kernel 11x11','',], 4, 'show')
plt.savefig("LPF_P.png")
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
    
filtered_imgs = {'A':[],'P':[]}
filter_images(3, hpf_3, imgs, filtered_imgs)   
filter_images(5, hpf_5, imgs, filtered_imgs)   
filter_images(7, hpf_7, imgs, filtered_imgs)   

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-alta para diferentes kernels - Amplitude")
plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs['A'], 4, 3, ['','Kernel 3x3','','','Kernel 7x7','','','Kernel 11x11','',], 4, 'show')
plt.savefig("HPF_A.png")
plt.close('all')

plt.figure(1, (16,9))
plt.suptitle("Filtro passa-alta para diferentes kernels - Phase")
plot_images(imgs, 4, 3, ['','Original',''], 1, 'new')
plot_images(filtered_imgs['P'], 4, 3, ['','Kernel 3x3','','','Kernel 7x7','','','Kernel 11x11','',], 4, 'show')
plt.savefig("HPF_P.png")
plt.close('all')

#%%