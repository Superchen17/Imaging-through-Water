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
    middleFonts2data.mat (mono)
    expdata_large.mat (mono)
    expdata_small.mat (mono)
    expdata_tiny.mat (mono)
'''

#Main method 
from Functions import loadMonoImage, processStack
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat  
import warnings
    
def main(fileName, saveFile=0):
    #Warning supprssion (Mostly CUDA warnings)
    warnings.filterwarnings('ignore')
    
    #Data loading
    Directory = "Datasets/"
    print(Directory+fileName)    
    frames = loadMonoImage(Directory+fileName) 
    
    #Model execution
    print("Restoration algorithm started...") 
    tic = time.time()    
    output = processStack(frames)
    toc = time.time() 
    
    #Normalize and save
    if(np.max(output) > 255):
        output = output*255/np.max(output)
    if(int(saveFile)==1):
        savemat(str(fileName)+'_Mono.mat', {'recon':output})    
    print("Algorithm processing time", round(toc-tic, ndigits=2), "seconds")
    
    #Output display
    '''
    plt.subplot(1,2,1)
    plt.imshow(frames[:,:,0],cmap='gray')
    plt.subplot(1,2,2)
    '''
    plt.imshow(output, cmap="gray")
    plt.show()      
    
import sys 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])