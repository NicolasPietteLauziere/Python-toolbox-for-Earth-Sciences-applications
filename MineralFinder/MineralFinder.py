# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:04:37 2021

@author: Nicolas Piette-Lauzière, PhD candidate, UBC Okanagan
Script to load chemical maps and create composite images with elements are
rgb counts. Chemical maps need to be GRAYSCALE 16bit bmp files 
named by elements Ca.bmp, P.bmp, etc.
"""
#-----------------------------------------------------------------------------
# Function definition
#-----------------------------------------------------------------------------
def loadMaps(inpath):
    """
    inpath is th path to a file containing 16bit tiff images of 
    Y, Ca, Zr, P

    returns a dataframe with each element as a column and the shape 
    of the images
    """
    
    #Ca
    ipath=os.path.join(inpath+"\\Ca.bmp")
    img = cv2.imread(ipath,-1) # load an imagem MUST BE GRAYSCALE 16bit
    #img = cv2.GaussianBlur(img,(kernel_size, kernel_size), x_sigma)
    # reshape the image to a 2D array of pixels and 1 greyscale value
    print(img.shape)
    img = img[:,:,1]
    pixels_value = img.reshape((-1, 1))
    pixels = np.empty(len(pixels_value))
    pixels = pixels_value
    #pixels = np.append(pixels,pixels_value,axis=1)
    print('Ca')
    
    #P
    ipath=os.path.join(inpath+"\\P.bmp")
    img = cv2.imread(ipath,-1) # load an imagem MUST BE GRAYSCALE 16bit
    #img = cv2.GaussianBlur(img,(kernel_size, kernel_size), x_sigma)
    # reshape the image to a 2D array of pixels and 1 greyscale value
    img = img[:,:,1]
    pixels_value = img.reshape((-1, 1))
    pixels = np.append(pixels,pixels_value,axis=1)
    print('P')
 
    #Y
    ipath=os.path.join(inpath+"\\Y.bmp")
    img = cv2.imread(ipath,-1) # load an imagem MUST BE GRAYSCALE 16bit
    #img = cv2.GaussianBlur(img,(kernel_size, kernel_size), x_sigma)
    # reshape the image to a 2D array of pixels and 1 greyscale value
    img = img[:,:,1]
    pixels_value = img.reshape((-1, 1))
    pixels = np.append(pixels,pixels_value,axis=1)
    print('Y')

    #Zr
    ipath=os.path.join(inpath+"\\Zr.bmp")
    img = cv2.imread(ipath,-1) # load an imagem MUST BE GRAYSCALE 16bit
    #img = cv2.GaussianBlur(img,(kernel_size, kernel_size), x_sigma)
    # reshape the image to a 2D array of pixels and 1 greyscale value
    img = img[:,:,1]
    pixels_value = img.reshape((-1, 1))
    pixels = np.append(pixels,pixels_value,axis=1)
    print('Zr')

    #Si
    ipath=os.path.join(inpath+"\\Si.bmp")
    img = cv2.imread(ipath,-1) # load an imagem MUST BE GRAYSCALE 16bit
    #img = cv2.GaussianBlur(img,(kernel_size, kernel_size), x_sigma)
    # reshape the image to a 2D array of pixels and 1 greyscale value
    img = img[:,:,1]
    pixels_value = img.reshape((-1, 1))
    pixels = np.append(pixels,pixels_value,axis=1)
    print('Si')

    #pixels = np.delete(pixels, 0,1)
    element = ['Ca','P','Y','Zr','Si']
    df = pd.DataFrame(pixels, columns=element)
    x,y = img.shape
    return(df, x, y)
    return(pixels)

def rgb(R, G, B, x, y):
    """
    r,g,b are channels for red, green, blue from a dataframe column
    x, y is the shape of the input png
    """
    # Plot RGB composite image with Y-P-Ca channels

    r=array(R)
    g=array(G)
    b=array(B)

    r=(r-min(r))/max(r)*255
    g=(g-min(g))/max(g)*255
    b=(b-min(b))/max(b)*255


    #
    rgb = np.dstack((r.tolist(), g.tolist(), b.tolist()))
    rgb = rgb.reshape(x,y,3)
    return(rgb)

#-----------------------------------------------------------------------------
#Main script
#-----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inpath = "C:\\Users\\Nicolas\\Documents\\GitHub\\MineralFinder\\input"
outpath = "C:\\Users\\Nicolas\\Documents\\GitHub\\MineralFinder\\output"

df,x,y = loadMaps(inpath)

# For example, plot apatite with Ca and P maps as red and green - Si blue for background
image = rgb(df.Ca, df.P, df.Si, x, y)

# Displaying the image  
plt.imshow(np.uint8(image))
plt.show()
plt.axis('off')

ipath=os.path.join(outpath+"\\"+'rgb_YPSi.tif')
plt.savefig(ipath, bbox_inches='tight',dpi=1200)


