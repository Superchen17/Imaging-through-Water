'''
Shift Map Analysis for Image Restoration
Reference: Correcction of Distorted Underwater Images using Shift Map Analysis

Available datasets:
- mp4:
    Knife.mp4 (color/mono)
    Heater1.mp4 (color/mono)
    Heater2.mp4 (color/mono)
    Pool1.mp4 (color/mono)
    Pool2.mp4 (color/mono)
    Pool3.mp4 (color/mono)
- mat:
    expdata_brick.mat (mono)
    expdata_checkboard.mat (mono)
    expdata_middle.mat (mono)
    expdata_large.mat (mono)
    expdata_small.mat (mono)
    expdata_tiny.mat (mono)
'''

from Functions import loadColorImage, processStack
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from scipy.io import savemat
from multiprocessing import Pool
import warnings

def main(fileName, saveFile):
    #Warning supprssion (Mostly CUDA warnings)    
    warnings.filterwarnings('ignore')    
    
    #Data loading
    Directory = "Datasets/"
    print(Directory+fileName)
    
    framesR, framesG, framesB = loadColorImage(Directory+fileName)
    frames_x, frames_y, frames_z = np.shape(framesR)
    original = np.zeros((frames_x,frames_y,3), "uint8")
    output = np.zeros((frames_x,frames_y,3), "uint8")
    
    original[:,:,0] = framesR[:,:,0]
    original[:,:,1] = framesG[:,:,0]
    original[:,:,2] = framesB[:,:,0]
    original = Image.fromarray(original)
    
    #Model execution
    print("Restoration algorithm started...")    
    tic = time.time()    
    pool = Pool(processes=3)
    temp = pool.map(processStack,[framesR,framesG,framesB])   
    pool.close()
    toc = time.time()               
    
    #Normalize and save
    output[:,:,0] = temp[0] 
    output[:,:,1] = temp[1]  
    output[:,:,2] = temp[2]     
    if(np.max(output) > 255):
        output = output*255/np.max(output)      
    if(int(saveFile)==1):
        savemat(str(fileName)+'_Color.mat', {'recon':output})        
    output = Image.fromarray(output)  
      
    print("Algorithm processing time", round(toc-tic, ndigits=2), "seconds")
    
    #Output display
    plt.subplot(1,2,1)
    plt.imshow(original,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(output,cmap="gray")
    plt.axis('off')        
    plt.show()
    
    return output

import sys 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])