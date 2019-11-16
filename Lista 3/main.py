#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:21:48 2019

@author: bruno
"""

import cv2
import numpy as np

def print_rotation_deg(rvecs):
    print(f''' rotation: [
                 X: {180*rvecs[0][0]/np.pi}               
                 Y: {180*rvecs[0][1]/np.pi}               
                 Z: {180*rvecs[0][2]/np.pi}                       
            ]              
        ''')
        
def print_translation(tvecs):
    print(f''' translation: [
                 X: {tvecs[0][0]}               
                 Y: {tvecs[0][1]}               
                 Z: {tvecs[0][2]}                       
            ]              
        ''')

cap = cv2.VideoCapture(0)

chessboard_size = (8,6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((np.multiply(*chessboard_size),3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

while cv2.waitKey(1) != ord('q'):
    _, frame = cap.read()   
    
    if frame is None: break

    frame = cv2.resize(frame, (320,240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7)
    ret, corners = cv2.findChessboardCorners(thresh, chessboard_size, None)
    if ret:        
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1],None,None)
        
        print("############################################")
        print_rotation_deg(rvecs)
        print_translation(tvecs)
        print("********************************************")

    cv2.imshow('frame', frame)

cv2.destroyAllWindows()
cap.release()

#%%