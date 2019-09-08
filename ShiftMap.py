'''
Shift Map Analysis for Image Restoration
Reference: Correcction of Distorted Underwater Images using Shift Map Analysis

Available datasets:
- mp4:
    Knife.mp4
    Heater1.mp4
    Heater2.mp4
    Pool1.mp4
    Pool2.mp4
    Pool3.mp4
- mat:
    expdata_brick.mat
    expdata_checkboard.mat
    middleFonts2data.mat
    expdata_large.mat
    expdata_small.mat
    expdata_tiny.mat
'''

#Imports
import cv2
import scipy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

#Unchangeable global variables
frames = []
frames_x = 0
frames_y = 0
frames_z = 0

#Changeable global variables
pixelMultiplier = 255
upscaler = 1

# -1. Load image frames with OpenCV
from scipy.misc import imresize
from skimage.transform import rescale, resize, downscale_local_mean
def loadImage(fileName):
    global frames, frames_x, frames_y, frames_z, pixelMultiplier, upscaler
    # Reading MATLAB .mat file
    if(fileName.endswith(".mat")):
        frames = loadmat(fileName, appendmat=False).get('frames')   
        frames = frames[:,:,0:60]*pixelMultiplier 
        image = rescale(frames[:,:,0], upscaler, anti_aliasing=True)    
        frames_new = np.zeros((np.shape(image)[0],np.shape(image)[1],np.shape(frames)[2]))
        for i in range(np.shape(frames)[2]):
            image = rescale(frames[:,:,i], upscaler, anti_aliasing=True)
            frames_new[:,:,i] = image        
        frames = frames_new  
        
    # Reading .mp4 file    
    elif(fileName.endswith(".mp4")):
        cap = cv2.VideoCapture(fileName)        
        frames_x = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = np.zeros((frames_x,frames_y,frames_z))
        fc = 0
        ret = True
        while (fc < frames_z  and ret):
            ret, frame = cap.read()
            frames[:,:,fc] = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            fc += 1   
        cap.release()            
        image = rescale(frames[:,:,0], 0.25, anti_aliasing=True)
        frames_new = np.zeros((np.shape(image)[0],np.shape(image)[1],np.shape(frames)[2]))
        for i in range(np.shape(frames)[2]):
            image = rescale(frames[:,:,i], 0.25, anti_aliasing=True)
            frames_new[:,:,i] = image        
        frames = frames_new     
    
    frames_x,frames_y,frames_z = np.shape(frames) 
    print("Image loaded:", frames_x, frames_y, frames_z)   

def forWardDewarping(input, shiftRuleX, shiftRuleY):
    global frames_x, frames_y
    output = np.zeros_like(input)
    for i in range (frames_x):
        for j in range (frames_y):
            xNew = np.int0(i+shiftRuleX[i,j])
            yNew = np.int0(j+shiftRuleY[i,j])
            if(xNew >= frames_x):
                xNew = frames_x-1
            if(yNew >= frames_y):
                yNew = frames_y-1
            if(xNew < 0):
                xNew = 0
            if(yNew < 0):
                yNew = 0    
            output[xNew,yNew] = input[i,j]
    return output

def backDewarping(input, shiftRuleX, shiftRuleY):
    global frames_x, frames_y
    output = np.zeros_like(input)
    for i in range (frames_x):
        for j in range (frames_y):
            xNew = np.int0(i+shiftRuleX[i,j])
            yNew = np.int0(j+shiftRuleY[i,j])
            if(xNew >= frames_x):
                xNew = frames_x-1
            if(yNew >= frames_y):
                yNew = frames_y-1
            if(xNew < 0):
                xNew = 0
            if(yNew < 0):
                yNew = 0    
            output[i,j] = input[xNew,yNew]
    return output 

# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (cooef1 + cooef2)/2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    elif (method == 'add'):
        cooef = cooef1+cooef2
        cooef = cooef*max(np.max(cooef1),np.max(cooef2))/np.max(cooef) 
    elif (method == 'merge'):
        cooef = sharpFuse(cooef1,cooef2)    
    else:
        cooef = []
    return cooef

import pywt
def waveletFuse(input1,input2):
    ## Fusion algo   
    # First: Do wavelet transform on each image
    wavelet = 'bior1.1'
    cooef1 = pywt.wavedec2(input1, wavelet)
    cooef2 = pywt.wavedec2(input2, wavelet)
    
    # Second: for each level in both image do the fusion according to the desire option
    fusedCooef = []
    for i in range(len(cooef1)):    
        # The first values in each decomposition is the apprximation values of the top level      
        if(i == 0):  
            fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],'mean')) #LL            
        else:
            # For the rest of the levels we have tupels with 3 coefficents                
            c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],'mean') #LH 
            c2 = fuseCoeff(cooef1[i][1],cooef2[i][1],'mean') #HL
            c3 = fuseCoeff(cooef1[i][2],cooef2[i][2],'mean') #HH                                              
            fusedCooef.append((c1,c2,c3))                
    # Third: After we fused the cooefficent we need to transfor back to get the image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)        
    return fusedImage

def denoise(input): 
    input = np.float32(input)
    ret, mask = cv2.threshold(input,1,255, cv2.THRESH_BINARY_INV)
    mask = mask.astype('uint8')
    input = cv2.inpaint(input,mask,3,cv2.INPAINT_TELEA)     
    return input 

#Main method 
import math
def main(argv):
    global frames, frames_x, frames_y, frames_z
    
    Directory = "Datasets/"
    fileName = argv
    print(Directory+fileName)
    
    tic = time.time()
    loadImage(Directory+fileName) # call for loading image (step 1)   
    original = frames[:,:,0]
    
    while(frames_z>1):
        frames_new = np.zeros((frames_x,frames_y,math.ceil(frames_z/2)))
        for i in range(0, frames_z-1,2):
            print("Iteration", i, "/", frames_z-1)
            reference = frames[:,:,i]
            target = frames[:,:,i+1]
            
            flow_rt = cv2.calcOpticalFlowFarneback(target,reference,None,0.5,3,15,3,5,1.2,0)/2
            warped_rt = backDewarping(reference, flow_rt[:,:,1], flow_rt[:,:,0])
            flow_tr = cv2.calcOpticalFlowFarneback(reference,target,None,0.5,3,15,3,5,1.2,0)/2                                    
            warped_tr = backDewarping(target, flow_tr[:,:,1], flow_tr[:,:,0]) 
            
            warped = waveletFuse(warped_rt,warped_tr)
            warped = cv2.resize(warped,(frames_y,frames_x))  
            frames_new[:,:,i//2] = warped
            
            if(i == frames_z-3):
                frames_new[:,:,(i//2)+1] = frames[:,:,-1]
            
        frames = frames_new
        frames_x, frames_y, frames_z = np.shape(frames)
        
    output = frames[:,:,0]
    toc = time.time()
    
    print("Processing time", toc-tic, "seconds")
    plt.subplot(1,2,1)
    plt.imshow(original, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(output, cmap="gray")
    plt.show()    

import sys 
if __name__ == "__main__":
    main(sys.argv[1])