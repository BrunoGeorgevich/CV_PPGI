#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:55:38 2020

@author: bruno
"""
#%% READING LIBS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skimage.feature import hog
from sklearn.svm import SVC
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import pickle
import cv2

#%% GENERATING TRAIN AND TEST CSVS
df = pd.read_csv('annotation.csv')

df_train, df_test = train_test_split(df)

with open('train.csv', 'w') as f:
    f.write(df_train.to_csv(index=False))
    
with open('test.csv', 'w') as f:
    f.write(df_test.to_csv(index=False))

#%% GENERATING HOG CSV FILE

df = pd.read_csv('train.csv')

while_condition = True

def time_elapsed(func):
    def function_wrapper(row):
        before = time()
        func(row)
        after = time()
        print(f"TIME ELAPSED :: {after - before:.2f}")
    return function_wrapper

def extract_hog(row):
    im = cv2.imread(row['filename'])
    x,y,w,h = row[['x','y','w','h']]  
    
    x_1 = int((x - w/2)*im.shape[1])
    x_2 = int((x + w/2)*im.shape[1])
    y_1 = int((y - h/2)*im.shape[0])
    y_2 = int((y + h/2)*im.shape[0])
    
    if (x_2 - x_1) < 100: return None
    if (y_2 - y_1) < 100: return None
    
    cropped = im[y_1:y_2, x_1:x_2]
    cropped = cv2.resize(cropped, (100, 100))
    cropped_hog = hog(cropped, multichannel=True)
    
    return cropped_hog

num_of_samples = 100000

with open('train_hog.csv', 'w') as f:
    f.write('filename,x,y,w,h,class,' + ','.join([f'item_{i}' for i in range(8100)]) + '\n')
    for idx, row in tqdm(df.sample(num_of_samples).iterrows(), total=num_of_samples):
        ret = extract_hog(row)
    
        if ret is None: continue
        
        filename,x,y,w,h,c = row[['filename','x','y','w','h','class']]
    
        f.write(f'{filename},{x},{y},{w},{h},{c},' + ','.join([f'{i}' for i in ret]) + '\n')
#%% SPLIT DATA

df = pd.read_csv('train_hog.csv')
X = df[[f'item_{i}' for i in range(8100)]]
y = df['class']
classes = list(set(y))

X_train, X_test, y_train, y_test = train_test_split(X,y)

#%% TRAIN CLASSIFIER

clfs = {}

for c in classes:
    print(f'TRAINING CLASSIFIER OF THE CLASS {c}')
    clf = SVC()    
    clf.fit(X_train, y_train.map(lambda it: 1 if it == c else 0))
    clfs[c] = clf

#%% SAVE MODEL

for c in classes:
    clf = clfs[c]
    with open(f'model_c{c}.clf','wb') as f:
        s = pickle.dumps(clf)
        f.write(s)
    
#%% READ MODEL

clfs = {}
classes = [i for i in range(7)]
for c in classes:
    clf = pickle.load(open(f'model_c{c}.clf','rb'))
    clfs[c] = clf

#%% TEST MODEL

for c in classes:
    clf = clfs[c]
    print(f'TESTING CLASSIFIER OF THE CLASS {c}')
    print(clf.score(X_test, y_test.map(lambda it: 1 if it == c else 0)))

#%%

def check_results(result):
    binary_result = list(map(lambda it: it[1] == 0, result))
    ret = np.sum(binary_result)
    if ret == 7: return -1
    if ret == 6: return np.argmin(binary_result) 
    if ret < 6: return -1

def predict(cropped_hog, clfs, one_class=False, choosed_class=-1):    
    result = []
    if one_class:    
        clf = clfs[choosed_class]
        return clf.predict(cropped_hog)
    else:
        for c in clfs.keys():
            clf = clfs[c]
            result.append([c, clf.predict(cropped_hog)])
        return check_results(result)
    

def resize_rounded(im):
    h,w,c = im.shape
    
    w_rounded = int(np.round(w/100)*100)
    h_rounded = int(np.round(h/100)*100)
    
    im = cv2.resize(im, (w_rounded, h_rounded))
    return im

def generate_sliding_window(im, window_dim, step=50):
    w_h, w_w = window_dim
    im_h, im_w, im_c = im.shape
    for y in range(0, im_h - w_h, step):
        for x in range(0, im_w - w_w, step):
            yield (x, y), (x+w_w, y+w_h), im[y : y+w_h, x : x+w_w]

test_df = pd.read_csv('test.csv')
filenames = list(set(test_df['filename']))

filename = filenames[0]

im = cv2.imread(filename)
im = resize_rounded(im)

im_h, im_w, _ = im.shape

window_dim = (50, 50)
w_h, w_w = window_dim

step = 10

print(int((im_h - w_h)/step), int((im_w - w_w)/step))
predicted_matrix = np.zeros((int((im_h - w_h)/step), int((im_w - w_w)/step)))

for pt1, pt2, cropped in generate_sliding_window(im, window_dim, step):
    im_cpy = im.copy()
    cv2.rectangle(im_cpy, pt1, pt2, (0, 0, 255), 3)
    cv2.imshow('im', im_cpy)
    
    cropped = cv2.resize(cropped,(100, 100))
    cropped_hog = hog(cropped, multichannel=True).reshape(1, -1)
    predicted_class = predict(cropped_hog, clfs, True, 0)

    x, y = pt1
    predicted_matrix[int(y/step),int(x/step)] = predicted_class

    if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()

#%% DRAW PREDICTED MATRIX

pm_h, pm_w = predicted_matrix.shape

im_cpy = im.copy()

for y in range(pm_h):
    for x in range(pm_w):
        pt1 = (int(x*step), int(y*step))
        pt2 = (int(x*step) + w_w, int(y*step) + w_h)
        predicted_class = predicted_matrix[y,x]
        if predicted_class == 1:
            cv2.rectangle(im_cpy, pt1, pt2, (0, 0, 255), 3)
        
cv2.imshow('im', im_cpy)
cv2.waitKey(0)
cv2.destroyAllWindows()