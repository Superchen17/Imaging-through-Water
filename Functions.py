'''
Function class
'''

'''
Imports
'''
import cv2
import scipy
import numpy as np
import pywt
import math
import queue
from scipy.io import loadmat
from skimage.transform import rescale, resize, downscale_local_mean

'''
Grayscale image loader
'''
def loadMonoImage(fileName):
    frames = []
    frames_x = 0
    frames_y = 0
    frames_z = 0    
    pixelMultiplier = 255
    upscaler = 1
    
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
    return frames

'''
Color image loader
'''
def loadColorImage(fileName):    
    framesR = [];
    framesG = [];
    framesB = [];
    frames_x = 0;
    frames_y = 0;
    frames_z = 0;
    rescaler = 0.25
    
    cap = cv2.VideoCapture(fileName)        
    ret, frame = cap.read()
    f = rescale(frame[:,:,0], rescaler, anti_aliasing=True)
    frames_x,frames_y = np.shape(f)
    frames_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()            
    
    framesR = np.zeros((frames_x,frames_y,frames_z))
    framesG = np.zeros((frames_x,frames_y,frames_z))
    framesB = np.zeros((frames_x,frames_y,frames_z))
    
    cap = cv2.VideoCapture(fileName)                
    fc = 0
    ret = True
    while (fc < frames_z  and ret):
        ret, frame = cap.read()
        frameBTemp, frameGTemp, frameRTemp = cv2.split(frame)    
        framesB[:,:,fc] = rescale(frameBTemp, rescaler, anti_aliasing=True)  
        framesG[:,:,fc] = rescale(frameGTemp, rescaler, anti_aliasing=True)   
        framesR[:,:,fc] = rescale(frameRTemp, rescaler, anti_aliasing=True)                  
        fc += 1   
    cap.release()        

    frames_x,frames_y,frames_z = np.shape(framesR) 
    print("Image loaded:", frames_x, frames_y, frames_z)   
    return framesR, framesG, framesB

'''
Gap filling for forward mapping
'''
def denoise(input): 
    input = np.float32(input)
    ret, mask = cv2.threshold(input,1,255, cv2.THRESH_BINARY_INV)
    mask = mask.astype('uint8')
    input = cv2.inpaint(input,mask,3,cv2.INPAINT_TELEA)     
    return input 

'''
Forward mapping according to warping field
'''
def forWardDewarping(input, shiftRuleX, shiftRuleY):
    frames_x,frames_y = np.shape(input)
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
    output = denoise(output)
    return output

'''
Backward mapping according to warping field
'''
def backDewarping(input, shiftRuleX, shiftRuleY):
    frames_x, frames_y = np.shape(input)
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

'''
Wavelet coefficient fusion rule
'''
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

'''
Image fusion with wavelet coefficient decomposition
'''
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

'''
Adjacent image fusion procedure
'''
def processStack(frames, template, mode):
    frames_x, frames_y, frames_z = np.shape(frames)
    
    if(mode == 'mono'):
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
    elif(mode == 'color'):
        while(frames_z>1):
            frames_new = np.zeros((frames_x,frames_y,math.ceil(frames_z/2)))
            temp_new = np.zeros((frames_x,frames_y,math.ceil(frames_z/2)))
            
            for i in range(0, frames_z-1,2):
                print("Iteration", i, "/", frames_z-1)
                reference = template[:,:,i]
                target = template[:,:,i+1]
                frame_rt = frames[:,:,i]
                frame_tr = frames[:,:,i+1]                
                
                flow_rt = cv2.calcOpticalFlowFarneback(target,reference,None,0.5,3,15,3,5,1.2,0)/2
                template_rt = backDewarping(reference, flow_rt[:,:,1], flow_rt[:,:,0])                
                warped_rt = backDewarping(frame_rt, flow_rt[:,:,1], flow_rt[:,:,0])
                flow_tr = cv2.calcOpticalFlowFarneback(reference,target,None,0.5,3,15,3,5,1.2,0)/2     
                template_tr = backDewarping(target, flow_tr[:,:,1], flow_tr[:,:,0])                 
                warped_tr = backDewarping(frame_tr, flow_tr[:,:,1], flow_tr[:,:,0]) 
                
                warped_frame = waveletFuse(warped_rt,warped_tr)
                warped_frame = cv2.resize(warped_frame,(frames_y,frames_x))  
                warped_temp = waveletFuse(template_rt,template_tr)
                warped_temp = cv2.resize(warped_temp,(frames_y,frames_x))                  
                frames_new[:,:,i//2] = warped_frame
                temp_new[:,:,i//2] = warped_temp
                
                if(i == frames_z-3):
                    frames_new[:,:,(i//2)+1] = frames[:,:,-1]
                    temp_new[:,:,(i//2)+1] = template[:,:,-1]                   
                
            frames = frames_new
            template = temp_new
            frames_x, frames_y, frames_z = np.shape(frames)        
        output = frames[:,:,0]            
    return output

