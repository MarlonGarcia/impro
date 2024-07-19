# -*- coding: utf-8 -*-
'''
Image Functions Library

Library for image pre-processing, with functions to handle and prepare images
for machine learning and image processing. This library accounts for functions
to: load and plot a group of images, pre-processing, choose ROI regions (may be
polygonal), choose points, get image properties, align and transform images
(including rotate, scale, etc.), filter signals and images (2D data), among
others.


OBS: some functions use the 'pynput' and 'windsound' libraries, which may be
difficult to install and do not works on non-windows platforms. Comment these
library imports if there are problems during installation or loading.

@author: Marlon Rodrigues Garcia
@school: School of Engineering, Campus of Sao Joao da Boa Vista
@instit.: Sao Paulo State University (Unesp)
@contact: marlon.garcia@unesp.br
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import stats
import time
import ctypes
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import tifffile as tiff       # To import ".lsm" images
from random import shuffle
from scipy import fftpack     # to apply FFT and iFFT
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from scipy.ndimage import center_of_mass
from skimage.measure import shannon_entropy
from pynput import keyboard   # It does not worked on Google Colab
import winsound               # To perform 'beep' sounds, only works on Windows



def list_folders(directory):
    '''
    Function to list all folders inside a directory

    Parameters
    ----------
    directory : string
        A string with the directory. Example: 'C:/Users/User/My Drive/Data/'.

    Returns
    -------
    folders : list
        A list with all folders inside the directory.
    '''
    # First we eliminate problems of commands with "\" (e.g. "\n", "\t", etc.)
    bytes_string = directory.encode('unicode_escape')
    directory = bytes_string.decode('utf-8')
    # Then we verify if the itens inside 'directory' are really folders
    folders = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            folders.append(item)
    return folders



def list_images(directory):
    '''
    Function to list all images inside a folder

    Parameters
    ----------
    directory : string
        A string with the directory. Example: 'C:/Users/User/My Drive/Data/'.

    Returns
    -------
    folders : list
        A list with all the images inside the folder.
    '''
    # First we eliminate problems of commands with "\" (e.g. "\n", "\t", etc.)
    bytes_string = directory.encode('unicode_escape')
    directory = bytes_string.decode('utf-8')
    # Defining which image to consider
    extensions = ['.jpg', '.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.gif',
                  '.GIF', '.tif', '.TIF', '.tiff', '.TIFF', '.heic', '.HEIC',
                  '.heif', '.HEIF', '.psd', '.PSD', '.raw', '.RAW', '.bmp',
                  '.BMP', '.lsm', '.LSM']
    # Then we verify if the itens inside 'directory' are really images
    image_names = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            if any(item.lower().endswith(ext) for ext in extensions):
                image_names.append(item)
    return image_names



def load_gray_images(folder, **kwargs):
    """Loading grayscale images from 'folder'
    
    This function loads all the images from 'folder' directory, in grayscale
    format, and store them in the variable 'I'.
    
    if colormap = -1, no colormap is assigned
    if colormap = cv2_COLORMAP_OPTION, (being option = hsv, jet, parula, etc),
    or a colormap reference number (0, 1, 2, etc), the colormap chosen option
    is assigned.
    """
    I = []
    # Standard colormap is grayscale
    colormap = kwargs.get('colormap')
    flag1 = cv2.IMREAD_GRAYSCALE
    # Listing images (usefull for when online drives put files in the folder)
    image_names = list_images(folder)
    if colormap == -1 or colormap == None:
        for filename in image_names:
            img = cv2.imread(os.path.join(folder, filename), flag1)
            if img is not None:
                I.append(img)
    else:
        for filename in image_names:
            img = cv2.imread(os.path.join(folder, filename), flag1)
            if img is not None:
                img2 = cv2.applyColorMap(img, colormap)
                I.append(img2)
    return I



def load_color_images(folder):
    """Loading colorful images from 'folder'
    
    This function load all colorful images from 'folder' in variable I.
    """
    I = []
    flag1 = cv2.IMREAD_COLOR
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), flag1)
        if img is not None:
            I.append(img)
    return I



def plot_gray_images(I, n):
    '''Program to plot 'n' images from 'I' using 'opencv2'
    
    This program will plot 'n' images from variable the list 'I' (a list of 
    numpy arrays). Press 'ESC' for close all the windows, or another key to 
    mantain the windows.
    '''
    I1 = np.asarray(I)
    for count in range(0,n):
        # We use cv2.WINDOW_AUTOSIZE to allow changes in image size.
        # If we use  WINDOW_NORMAL, we cannot change the image size (maximize).
        flag3 = cv2.WINDOW_NORMAL
        name = 'Figure ' + str(count+1)
        cv2.namedWindow(name, flag3)
        cv2.imshow(name, I1[count])
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # Waiting for 'ESC' key before close all the windows
        cv2.destroyAllWindows()



def plot_color_images(I, n):
    '''Program to plot 'n' color images from 'I' using 'opencv2'
    
    This program will plot 'n' color images from variable the list 'I' (a list
    of numpy arrays). Press 'ESC' for close all the windows, or another key to
    mantain the windows.
    '''
    I1 = np.asarray(I)
    for count in range(0,n):
        # We use cv2.WINDOW_AUTOSIZE to allow changes in image size.
        # If we use  WINDOW_NORMAL, we cannot change the image size (maximize).
        flag3 = cv2.WINDOW_NORMAL
        name = 'Figure ' + str(count+1)
        cv2.namedWindow(name, flag3)
        cv2.imshow(name, I1[count])
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # Waiting for 'ESC' key before close all the windows
        cv2.destroyAllWindows()



def plot_gray(I, name, colormap):
    '''Program to plot gray images with 'matplotlib' from the list 'I'
    
    I: input image (as a 'list' variable)
    name: window name
    colormap: a colormap name (pink, RdGy, gist_stern, flag, viridis, CMRmap)
    
    This program will plot gray images from the list in 'I', using matplotlib.
    '''
    if name is None:
        name = 'Figure'
    I1 = np.asarray(I)
    shape = I1.shape
    if len(shape) == 2:
        plt.figure(name)
        plt.imshow(I1, colormap)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    else:
        print('\nYour variable "I" is not a recognized image\n')



def plot_bgr(I,name):
    '''Program to plot BGR images with 'matplotlib' from the list 'I'
    
    This program will plot RGB images from the list in 'I', using matplotlib.
    '''
    if name is None:
        name = 'Figure'
    I1 = np.asarray(I)
    shape = I1.shape
    if len(shape) == 3:
        plt.figure(name)
        plt.imshow(I1[:,:,::-1])
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    else:
        print('\nYour variable "I" is not a recognized image\n')



def beep(**kwargs):
    ''' Function to make a beep
    beep(freq,duration)
    
    **freq: tone frequency, in hertz (preseted to 2500 Hz)
    **duration: tone duration, in miliseconds (preseted to 300 ms)
    '''
    freq = kwargs.get('freq')
    duration = kwargs.get('duration')
    if freq is None:
        freq = 2500  # Set Frequency To 2500 Hertz
    if duration is None:
        duration = 300  # Set Duration To 300 ms == 0.3 second
    numb = 5 # number of beeps
    for n in range(0, numb):
        time.sleep(0.0005)
        winsound.Beep(freq, duration)



def read_lsm(file_path):
    '''Reading and mounting images of '.lsm' extension from Zeiss microscope 

    Parameters
    ----------
    file_path : string
        Path/Directory to the '.lsm' file.

    Returns
    -------
    full_image : numpy.ndarray
        Full assembled image.
    '''
    # Reading image and its metadata
    with tiff.TiffFile(file_path) as tif:
        images = tif.asarray()
        metadata = tif.lsm_metadata
    
    # Transforming shape (Tiles, Channels, Width, Height) in (T, W, H, Chan.)
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Extracting the tile positions
    tile_positions = metadata['TilePositions']
    
    # Discovering how many tiles we have on the horizontal axis (x axis)
    tile_x = 1
    for n in range(1, len(tile_positions)):
        if tile_positions[n, 0] == tile_positions[0, 0]:
            tile_x = n
            break
    
    # Discovering how many tiles we have on the vertical axis (y axis)
    tile_y = 1
    value = tile_positions[0, 1]
    for n in range(1, len(tile_positions)):
        if tile_positions[n, 1] != value:
            tile_y += 1
            value = tile_positions[n, 1]
    
    # Mounting the full image
    shape = np.shape(images)
    full_image = np.zeros([shape[2]*tile_y, shape[1]*tile_x, 3], dtype='uint8')
    for x in range(tile_x):
        for y in range(tile_y):
            full_image[y*shape[2]:(y+1)*shape[2], shape[1]*x:shape[1]*(x+1), :] = images[x + tile_x*y, :, :, :]
    
    return full_image



def rotate2D(pts, cnt, ang):
    '''Rotating the points about a center 'cnt' by an ang 'ang' in radians.
    
    [pts_r] = rotate2D(pts, cnt, ang)
    
    '''
    return np.dot(pts-cnt,np.array([[ np.cos(ang),np.sin(ang)],
                                    [-np.sin(ang),np.cos(ang)]]))+cnt



class choose_points1(object):
    '''This is the class to help 'choose_points' function.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a 'plus symbol' in image).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new point added to 'points' .
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of the choose, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the 'choose_points1' class:
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose points"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # Defining thickness based on image size (to lines in image)
            if np.shape(image2)[0] > 400:
                radius = int(np.shape(image2)[1]/200)
            else:
                radius = 2
            
            # If at least 1 point was chosen, draw the points. We use
            # cv2.imshow to constantly show a new image with 
            # the chosen points.
            ### Next 'circle' is disabled, to does not show the current point:
            # image2 = cv2.circle(image2, self.current, radius=radius,
            #                     color=(222, 222, 252), thickness=radius-2)
            if (len(self.points) > 0):
                for n in range(0,len(self.points)):
                    cv2.circle(image2, self.points[n], radius=radius,
                               color=(232, 222, 222), thickness=radius-2)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            for n in range(0,len(self.points)):
                cv2.circle(image2, self.points[n], radius=radius,
                           color=(232, 222, 222), thickness=radius-2)
            cv2.imshow(window_name, image2)
        
        return image2, np.asarray(self.points), window_name    



def choose_points(image, **kwargs):
    '''This function return the local of chosen points.
    
    Example:
    -------
    [image_out, points] = choose_points(image, **cmap, **window_name, **show)
    
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the points remains printed.
    image_out: the output polygonal ROI
    points: the chosen vertices (numpy-array)
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose points in an image, and
    to get their positions.
    
    **The files with double asteristic are optional (**kwargs).    '''   
    choose_class = choose_points1()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1=3) or gray (img_type1=2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'policlass.run' actually run the program we need.
    [image2, points, window_name] = choose_class.run(image, cmap,
                                                   window_name, img_type, show)
    # If show = True, mantain the image ploted.
    if show != True:
        cv2.waitKey(500)    # Wait a little to user see the chosen points.
        cv2.destroyWindow(window_name)
    
    return image2, points



def flat2im(flat, shape):
    '''Converting a flat vector into an image. If the input flat vector has a
    shape of `N x C`, the output image will be of shape `H, W, C`, where `N =
    H * W` and C is the number of channels (1 for grayscale, 3 for RGB, or more
    for a multidimensinal image).
    
    Example:
    --------
    image = flat2im(flat, shape)
    
    Parameters
    ----------
    flat : numpy.ndarray
        Input flat vector of shape `N x C` where `N = height*width` and C is
        the number of channels (1 for grayscale, 3 for RGB, etc.)
    shape : tuple or list
        Tuple or list containing the height and the width of the image:
        `(height, width)`, for example `(2048, 1024)`
    '''
    # Transforming flat into a numpy.ndarray
    flat = np.asarray(flat)
    # If shape has more dimensions, extracting first 2
    shape = shape[0:2]
    # Verifying if user entered with wrong shapes for image, height or width
    if flat.shape[0] != shape[0] * shape[1]:
        raise ValueError(f'The flat vector does not have the correct size for {shape[0]} in height and {shape[1]} in width.')
    
    # Transforming the vector into an image after checking if it is in gray-
    # scale or multichannel (which stands for RGB or more channels)
    if len(flat.shape) == 1:
        image = flat.reshape(shape)
    else:
        image = flat.reshape((shape[0], shape[1], flat.shape[1]))
        
    return image



def im2flat(image, **kwargs):
    ''' Converting an image into a flat vector (can be RGB or multidimensional)
    
    Example:
    --------
    flat = imfun.im2flat(image)
    
    Parameters
    ----------
    image : numpy.ndarray (W, H, C)
        The image to be flatten. It could be a grayscale image, with (H, W)
        shape, an RGB image with (H, W, C) shape, or even a multidimensional
        image with multiple channels. Here, 'H' stands for higher, 'W' for
        width and 'C' for channels.
    
    **kwargs : 
        Additional arguments to control various options:
        
        - mask : numpy.ndarray
            When this mask is passed, only the pixels of `image` in the posi-
            tions where `mask` is non-zero will be considered to build the
            flat vector.

    Returns
    -------
    flat : numpy.ndarray
        A flatten array with a shape (H*W, C).'''
    
    # Getting the keyword arguments (**kwargs), if any
    mask = kwargs.get('mask')
    # Transforming image into 'numpy.ndarray'
    image = np.asarray(image)
    
    # If mask is multidimensional, transforming it into a grayscale image
    if mask is not None:
        mask = np.asarray(mask)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
    
    shape = image.shape
    if len(shape) == 2:
        if mask is not None:
            return image[mask!=0]
        else:
            return image.reshape((shape[0]*shape[1], 1))
    elif len(shape) == 3:
        if mask is not None:
            return image[mask!=0]
        else:
            return image.reshape((shape[0]*shape[1], shape[2]))
    else:
        raise ValueError(f'The input image has to be three dimensions, but {len(shape)} was passed.')



class im2label_class(object):
    '''This is the class helps 'im2label' function.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a 'plus symbol' in image).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new point added to 'points' .
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of the choose, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the 'choose_points1' class:
    def run(self, image, label, cmap, w, h, color, **kwargs):
        # Reading kwargs if any
        label_name = kwargs.get('label_name')
        # Stating a window_name for opencv
        if label_name:
            window_name = 'Choose a region for the ' + label_name + ' label'
        else:
            window_name = 'Choose a region for label ' + str(label)
        
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        if w > 500:
            thickness = int(w/500)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            # Creating the zoom in figure, we used 2*int because int(w/3) can
            # be different from 2*int(w/6).
            zoom_in = np.zeros([2*int(h/6), 2*int(w/6), 3], np.uint8)
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                # Writing lines in big figure
                cv2.polylines(image2, np.array([self.points]), False, color,
                              thickness)
                cv2.line(image2, self.points[-1], self.current, color,
                         thickness)
                if self.current[1]-int(h/6) < 0:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[int(h/6)-self.current[1]:,int(w/6)-self.current[0]:,:] = image2[0:self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[int(h/6)-self.current[1]:,0:int(w/6)+w-self.current[0],:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[int(h/6)-self.current[1]:,:,:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                elif self.current[1]+int(h/6) > h:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[0:int(h/6)+h-self.current[1],int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[0:int(h/6)+h-self.current[1],0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[0:int(h/6)+h-self.current[1],:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                else:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[:,int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[:,0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[:,:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                
                # Making a 'cross' signal to help choosing region
                zoom_in[int(h/6),int(4*w/30):int(19*w/120)] = np.uint8(zoom_in[int(h/6),int(4*w/30):int(19*w/120)]+125)
                zoom_in[int(h/6),int(21*w/120):int(6*w/30)] = np.uint8(zoom_in[int(h/6),int(21*w/120):int(6*w/30)]+125)
                zoom_in[int(4*h/30):int(19*h/120),int(w/6)] = np.uint8(zoom_in[int(4*h/30):int(19*h/120),int(w/6)]+125)
                zoom_in[int(21*h/120):int(6*h/30),int(w/6)] = np.uint8(zoom_in[int(21*h/120):int(6*h/30),int(w/6)]+125)

                # Transforming 'zoom_in' is a zoom (it is a crop right now)
                h_z, w_z = np.shape(zoom_in)[0],  np.shape(zoom_in)[1]
                zoom_in = cv2.resize(zoom_in[int(h_z/2)-int(h_z/4):int(h_z)-int(h_z/4), int(w_z/2)-int(w_z/4):int(w_z)-int(w_z/4)], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                image2[0:2*int(h/6),w-2*int(w/6):w]=zoom_in
                cv2.imshow(window_name, image2)
            else:
                if self.current[1]-int(h/6) < 0:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[int(h/6)-self.current[1]:,int(w/6)-self.current[0]:,:] = image2[0:self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[int(h/6)-self.current[1]:,0:int(w/6)+w-self.current[0],:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[int(h/6)-self.current[1]:,:,:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                elif self.current[1]+int(h/6) > h:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[0:int(h/6)+h-self.current[1],int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[0:int(h/6)+h-self.current[1],0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[0:int(h/6)+h-self.current[1],:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                else:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[:,int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[:,0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[:,:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                # Making a 'cross' signal to help choosing region
                zoom_in[int(h/6),int(4*w/30):int(19*w/120)] = np.uint8(zoom_in[int(h/6),int(4*w/30):int(19*w/120)]+125)
                zoom_in[int(h/6),int(21*w/120):int(6*w/30)] = np.uint8(zoom_in[int(h/6),int(21*w/120):int(6*w/30)]+125)
                zoom_in[int(4*h/30):int(19*h/120),int(w/6)] = np.uint8(zoom_in[int(4*h/30):int(19*h/120),int(w/6)]+125)
                zoom_in[int(21*h/120):int(6*h/30),int(w/6)] = np.uint8(zoom_in[int(21*h/120):int(6*h/30),int(w/6)]+125)

                # Transforming 'zoom_in' is a zoom (it is a crop right now)
                h_z, w_z = np.shape(zoom_in)[0],  np.shape(zoom_in)[1]
                zoom_in = cv2.resize(zoom_in[int(h_z/2)-int(h_z/4):int(h_z)-int(h_z/4), int(w_z/2)-int(w_z/4):int(w_z)-int(w_z/4)], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                image2[0:2*int(h/6),w-2*int(w/6):w]=zoom_in
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        cv2.destroyWindow(window_name)
        return self.points



def im2label(root, classes, **kwargs):
    '''Function to create label images (or masks) from images in a folder.
    
    This function creates another folder, with the same name as `root` plus the
    string "labels", and saves the label images in this folder with the same
    name of original images. Since labeling takes a lot of time, this function
    can also identifies which images were alredy labeled before starting. The
    final output image is scaled between 0 to 255, which can be changed by
    setting `scale=False`. (TODO:enhance document., say about classes+1 or background)
    
    Example:
    --------
    images = im2label(root, classes, show = True)
    
    Parameters
    ----------
    root : str
        Root directory where the images are located.
    
    classes : int
        The number of classes to choose.
    
    **kwargs : 
        Additional arguments to control various options:
        
        - save_images : bool (default: True)
            Choose if the label images will be saved in a folder. By default,
            the images are saved in a folder with the same name as `root` but
            adding 'labels' at the end of its name.
        
        - open_roi : str (default: None)
            If open_roi is not `None`, the algorithm chooses open regions (re-
            gions that end at the image boundary). If `open_roi` = 'above', the
            chosen region will be an open region above the selected area, the
            opposite happens if `open_roi` = 'below', with a region below the
            chosen points.
        
        - scale : bool (default: True)
            If True, the label images will be scaled between 0 and 256. The sa-
            ved labels/classes will be, e.g., 0, 127, and 255 in the case of 3
            classes. If scale=False, the images will be labeled with integer
            numbers, like 0, 1, and 2 in the case of 3 classes. In this case,
            the saved images might appear almost black in the folder.
        
        - label_names : list (default: None)
            A list of strings with the names of the labels to be shown during
            the interaction with the user.
        
        - cmap : int (cv2 colormap, default: None)
            Optional colormap to apply to the image.
        
        - show : bool (default: False)
            If True, shows the final image and its label until the user presses
            'ESC' or any key.
        
        - equalize : bool (default: False)
            If True, equalizes the grayscale image histogram.
        
        - color : tuple (default: (200, 200, 200))
            Enter a different color to color the working line (R, G, B) with
            values from 0 to 255.
    
    Return
    ------
    images : list
        A list with the labeled images, all of `numpy.ndarray` type.
    
    Mouse actions:
    - Left button: select a new point in the label;
    - Right button: end selection and finish or go to another label;
    - ESC: finish selection (close the algorithm).
    
    Notes:
    ------
    - When using `open_roi`, it is only possible to choose points from the left
      part of the image to the right.
    - The remaining unlabeled pixels will be set as background pixels (they
      will belong to the last label). If a label is chosen more than once, the
      last chosen label will be applied.
    - Images can be multidimensional (`[height, width, dimensions]`).
    '''
    # Adding 1 will allow the user to be able to choose the number of classes
    # he entered, and to consider an aditiona class for the background (zero)
    classes = int(classes+1)
    # With 'kwargs' we can define extra arguments as input.
    save_images = kwargs.get('save_images', True)
    cmap = kwargs.get('cmap')
    open_roi = kwargs.get('open_roi')
    show = kwargs.get('show')
    equalize = kwargs.get('equalize')
    color = kwargs.get('color')
    scale = kwargs.get('scale')
    label_names = kwargs.get('label_names')
    
    # If no color was chosen, choose gray:
    color = (200, 200, 200) if color==None else color

    # First, we create a folder with name ´root´+ ' labels', to save results,
    # but only if the user choose to save the images (if save_images==True):
    if save_images:
        os.chdir(root)
        os.chdir('..')
        basename = os.path.basename(os.path.normpath(root))
        # If folder has been already created, the use of try prevent error output
        try: os.mkdir(basename+' labels')
        except: pass
        os.chdir(basename+' labels')
        # Verify if folder has some already labaled images, if yes, skip the 
        # labeled ones
        image_names = list_images(root)
        if os.listdir(os.curdir):
            for name in os.listdir(os.curdir):
                if name in image_names:
                    image_names.remove(name)
    else:
        image_names = list_images(root)
    
    # Shuffling the names. Very important when all images in the dataset are
    # almost equal, like for OCT.
    shuffle(image_names)
    
    # Next few lines create colors to be shown to the user as labels. The while
    # loop is used to ensure that none of the labels are null or equal to any
    # other label. Each line of the 'colors' vector is the values of the RGB
    # channels (that will be multiplied by 127 to be in the range 0-255)
    equal_lines = True
    null_lines = True
    count = 0
    while equal_lines==True or null_lines==True:
        colors = np.random.randint(0, 3, size=(classes, 3))
        # Checking if there are equal lines
        equal_lines = np.any(np.triu(np.all(colors[:, None, :]==colors, axis=2), 1))
        # Checking if one of the lines is null
        null_lines = np.any(np.all(colors == 0, axis=1))
        if count > 50:
            break
        count += 1
    
    # Variable with labeled images to return from the function
    images = []
    # This for will iterate in each image in 'root' folder
    for image_name in image_names:
        image = cv2.imread(os.path.join(root, image_name), cv2.IMREAD_COLOR)
        # Equalizing histogram
        if equalize == True:
            cv2.equalizeHist(image, image)
        # First the label image will be a '-1' array
        label_image = np.full(image.shape, -1)
        # Image width and higher
        w = np.shape(image)[1]
        h = np.shape(image)[0]
        # Iterate in each label (except the last one, that is background)
        label = 1
        while label < classes:
            # The '.run' class gives the chosen points
            im2label = im2label_class()
            # If user choose names for labels, send the names to 'run' function
            if label_names:
                points = im2label.run(image, label, cmap, w, h, color,
                                      label_name=label_names[label-1])
            else:
                points = im2label.run(image, label, cmap, w, h, color)
            points = np.asarray(points)
            # If no points were chosen, gives the option to label the unchosen
            # points in the image as background
            if len(points)<1:
                q1 = 'Label unchosen points as background?'
                q2 = '\n\n\'No\' will quit application (an error will eventually raise).'
                # Answer: asw = 6 (yes), asw = 7 (no) 
                asw = ctypes.windll.user32.MessageBoxW(0,q1+q2,"Pergunta", 4)
                if asw == 6:
                    break
            # Creating a ROI to signaling the chosen region with '1'
            roi = np.zeros_like(image)
            # First we run when roi regions are closed (open_roi == None)
            if not open_roi:
                cv2.fillPoly(roi, [np.asarray(points)], (1, 1, 1))
            
            # If 'above' or 'below' are choosen, then run the following
            elif open_roi == 'above' or open_roi == 'below':
                # If ROI is 'above', concatenate the upper image part, but
                # if the user choose points near to the sides, concatenate
                # the side last side points. Same to 'below'.
                if points[0,0] > h - points[0,1]: # DOWN-X
                    if w - points[-1,0] > h - points[-1,1]: # DOWN-DOWN
                        points[0,1] = h
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0],[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = None
                    elif w - points[-1,0] > points[-1,1]: # DOWN-UP
                        points[0,1] = h
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = np.array([[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h]])
                    else: # DOWN-RIGHT
                        points[0,1] = h
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0],[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h]])
                            
                elif points[0,0] > points[0,1]: # UP-X
                    if w - points[-1,0] > h - points[-1,1]: # UP-DOWN
                        points[0,1] = 0
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[0,h],[0,0]])
                    elif w - points[-1,0] > points[-1,1]: # UP-UP
                        points[0,1] = 0
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = None
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h],[0,h],[0,0]])
                    else: # UP-RIGHT
                        points[0,1] = 0
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h],[0,h],[0,0]])
                else: # LEFT-X
                    if w - points[-1,0] > h - points[-1,1]: # LEFT-DOWN
                        points[0,0] = 0
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0],[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[0,h]])
                    elif w - points[-1,0] > points[-1,1]: # LEFT-UP
                        points[0,0] = 0
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = np.array([[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h],[0,h]])
                    else: # LEFT-RIGHT
                        points[0,0] = 0
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0],[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h],[0,h]])
                
                if start_points is not None:
                    points = np.concatenate((start_points,points), axis=0)
                cv2.fillPoly(roi, [np.asarray(points)], (1, 1, 1))
            else:
                raise ValueError('\nvariable \'open_roi\' has to be \'above\' or \'below\'')
            # Ask if the label was currectly chosen:
            if label_names:
                q1 = 'Was ' + label_names[label-1] + ' label correctly chosen?'
            else:
                q1 = 'Was label ' + str(label) + ' correctly chosen?'
            q2 = '\n\nSelect \'No\' to redo the labeling \n\n\'Yes\': to go forward..'
            # Answer: asw = 6 (yes), asw = 7 (no) 
            asw = ctypes.windll.user32.MessageBoxW(0,q1+q2,"Pergunta", 4)
            if asw == 6:
                # Writing 'label' chosen in 'label_image' variable
                label_image[roi==1] = label
                # If is interesting to only change where there are no other
                # labels assigned, use another condition as follows:
                # label_image[(roi==1) & (label_image==-1)] = label
                
                # Drawing the ROI in the original image
                mask = np.zeros_like(roi)
                mask[..., 0] = int(colors[label, 0]*255/2)
                mask[..., 1] = int(colors[label, 1]*255/2)
                mask[..., 2] = int(colors[label, 2]*255/2)
                roi2draw = roi*mask
                image = cv2.addWeighted(image, 1, roi2draw.astype('uint8'), 0.4, 0)
                
            # Ask if the user wants to choose more parts to the same label
            if label_names:
                q1 = 'Want to choose more points for the ' + label_names[label-1] + ' label?'
            else:
                q1 = 'Want to choose more points for label ' + str(label) + '?'
            q2 = '\n\nSelect \'No\' to continue.. \n\n\'Yes\': to select more labels.'
            # Answer: asw = 6 (yes), asw = 7 (no) 
            asw = ctypes.windll.user32.MessageBoxW(0, q1+q2,"Pergunta", 4)
            if asw == 7:
                label += 1
        
        # Assigning the last label as background (label = 0).
        label_image[label_image==-1] = 0
        label_image = np.array(label_image[..., 0], np.uint8)
        if scale:
            label_image = scale255(label_image)
        # If 'show' = True
        if show:
            fig, ax = plt.subplots(1,2)
            # To change from BGR (used by OpenCV) to RGB we used "::-1"
            ax[0].imshow(image[:,:,::-1])
            ax[0].axis('off')
            ax[0].set_title('Reference Image')
            ax[1].imshow(label_image, vmax=np.max(label_image), vmin=np.min(label_image), cmap = 'viridis')
            ax[1].axis('off')
            ax[1].set_title('Labeled Image')
            fig.tight_layout()
        # Adding final label image in 'label_images', if 'save_images=True'
        if save_images:
            cv2.imwrite(image_name, label_image)
        images.append(label_image)
    print('\nAll the images were labeled')
    
    return images



class improfile_class(object):
    '''This is a class to help improfile function (choose polygonal ROI)
    
    Read 'improfile' function for more informations.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a line from last vertexn)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new vertex, so we add this to 'points' 
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of drawing, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the polyroi function
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose a polygonal ROI"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        image2 = image_c.copy()
        if np.shape(image2)[0] > 350:
            thickness = int(np.shape(image_c.copy())[0]/350)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                cv2.polylines(image2, np.array([self.points]), False,
                              (200,200,200), thickness)
                cv2.line(image2, self.points[-1], self.current,
                         (200,200,200), thickness)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27 or len(self.points) > 2:
                self.done = True
        length = np.hypot(self.points[0][0]-self.points[1][0],
                          self.points[0][1]-self.points[1][1])
        
        X = np.array(np.linspace(self.points[0][0],self.points[1][0],int(length)), int)
        Y = np.array(np.linspace(self.points[0][1],self.points[1][1],int(length)), int)
        profile = image2[X, Y]
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            image3 = image.copy()
            cv2.polylines(image3, np.array([self.points]),False,(200,200,200))
        
        if show is not None:
            cv2.imshow(window_name, image3)
        
        return profile, self.points, window_name



def improfile(image, **kwargs):
    '''Find the profile of pixels intensity between two points in an image
    
    [profile, points_out] = improfile(image, **cmap, **window_name, **show)
    
    image: the input image
    points_in: input points to perform the profiling (if the points are already
    obtained).
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the chosen polygon drawn
    profile: the output profile chosen
    points_out: the chosen vertices
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose a polygonal ROI
    (region of interest) using the mouse. Use the left button to 
    choose te vertices and the right button to finish selection.
    
    **The arguments with double asteristic are optional (**kwargs).
    '''
    profileclass = improfile_class()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1 = 3) or gray (img_type1=2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'profileclass.run' actually run the program we need.
    [profile, points, window_name] = profileclass.run(image, cmap,
                                                  window_name, img_type, show)
    cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    cv2.destroyWindow(window_name)

    # to modify function:
    # # Next we find the profile. If points are already passed, just find the
    # # profile, if not, fint the points and profile with 'profileclass' class
    # if points_in is None:
    #     # 'profileclass.run' actually run the program we need.
    #     [profile, points, window_name] = profileclass.run(image, cmap,
    #                                                   window_name, img_type, show)
    #     cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    #     cv2.destroyWindow(window_name)
    # else:
    #     length = np.hypot(self.points[0][0]-self.points[1][0],
    #                       self.points[0][1]-self.points[1][1])
        
    #     X = np.array(np.linspace(self.points[0][0],self.points[1][0],int(length)), int)
    #     Y = np.array(np.linspace(self.points[0][1],self.points[1][1],int(length)), int)
    #     profile = image2[X, Y]
    
    return profile, np.asarray(points)



def scale255(image):
    '''
    Function to scale the image to the range [0,255]
    
    image_out = scale255(image)
    
    input:
    'image' (np.array): image to be rescaled
    
    output:
    image_out (np.array, np.uint8): rescaled image
    
    OBS: Image should be of shape (H, W, C), where H is the height, W is the
    width and C is the number of channels, for multichannel images.
    '''
    # Discovering if the image is grayscale or colorful
    if len(np.shape(image)) == 2:
        img_type = 'gray'
    # We use '>=3' to accounts for multichannel images
    elif len(np.shape(image)) >= 3:
        img_type = 'color'
    else:
        raise ValueError('Image has to be at least a 2D array, but '+
                         str(len(np.shape(image)))+'D array was given.')
    if img_type == 'gray':
        image = image - np.min(image)
        if np.max(image) != 0:
            image = image*(255/np.max(image))
    elif img_type == 'color':
        for n in range(len(np.shape(image))):
            image[:,:,n] = image[:,:,n] - np.min(image[:,:,n])
            if np.max(image[:,:,n]) != 0:
                image[:,:,n] = image[:,:,n]*(255/np.max(image[:,:,n]))
    else:
        raise ValueError('Image type was not recognized, see the'+
                         ' see the documentation for more information.')
    
    image = np.array(image, np.uint8)
    
    return image



class polyroi1(object):
    '''This is a class to help polyroi function (choose polygonal ROI)
    
    Read 'polyroi' function for more informations.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The choosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a line from last vertexn)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new vertex, so we add this to 'points' 
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of drawing, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the polyroi function
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose a polygonal ROI"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        if np.shape(image_c)[0] > 500:
            thickness = int(np.shape(image_c)[0]/500)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                cv2.polylines(image2, np.array([self.points]), False,
                              (200,200,200), thickness)
                cv2.line(image2, self.points[-1], self.current,
                         (255,255,255), thickness)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            image3 = image.copy()
            cv2.fillPoly(image3, np.array([self.points]), (255,255,255))
            image4 = np.zeros(np.shape(image3), np.uint8)
            cv2.fillPoly(image4, np.array([self.points]), (255,255,255))
        
        if show is not None:
            cv2.imshow(window_name, image3)
        return image4, self.points, window_name



def polyroi(image, **kwargs):
    '''Choose a polygonhal ROI with mouse
    
    [image2, points] = polyroi(image, **cmap, **window_name, **show)
    
    image: the input image
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the chosen polygon drawn
    image2: the output polygonal ROI
    points: the chosen vertices
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose a polygonal ROI
    (region of interest) using the mouse. Use the left button to 
    choose te vertices and the right button to finish selection.
    
    **The arguments with double asteristic are optional (**kwargs).
    '''
    policlass = polyroi1()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1 = 3) or gray (img_type1 =2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'policlass.run' actually run the program we need.
    [image2, points, window_name] = policlass.run(image, cmap,
                                                  window_name, img_type, show)
    cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    cv2.destroyWindow(window_name)
    
    return image2, points



class crop_image1(object):
    '''This is a class to help the 'crop_image' function
    
    Read 'crop_image' for more informations.
    '''
    def __init__(self):
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.points = []        # Points to crop the image (corners).
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # Update the mouse current position (to draw the working rectangle).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate the first and the second main corners of
        # the rectangle to be chosen(upper-left and lower-right corners), so we
        # add this points to the 'points'.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # If right buttom pressed, we start to draw the rectangl again.
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points = []
    
    # This function actually run the 'crop_image' function
    def run(self, image, cmap, window_name, img_type):
        if cmap is not None:
            if img_type == 2:
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        thickness = int(np.shape(image_c)[1]/500)
        
        # Loop to draw the rectangles while the user choose the final one.
        while(not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the working rectangle from
            # its upper-left or lower-right corner until the current mouse
            # position (working rectangle).
            if (len(self.points) > 0):
                if (len(self.points) == 1):
                    cv2.rectangle(image2, self.points[0], self.current,
                                  (200,200,200), thickness)
                    cv2.imshow(window_name, image2)
            # We creat k to exit the function if we pressee 'ESC' (k == 27)
            k = cv2.waitKey(50) & 0xFF
            if (len(self.points) > 1) or (k == 27):
                self.done = True
            # We have to use this for the case when right mouse button is pres-
            # sed. This is to stop shown the old erased rectangle (which we
            # erase with the right button click).
            if (len(self.points) == 0):
                cv2.imshow(window_name, image_c)
            
        cv2.destroyWindow(window_name)
        # When leaving loop, draw the final rectangle IF at least two
        # points were chosen.
        if (len(self.points) > 1):
            image3 = image.copy()
            cv2.rectangle(image3, self.points[0], self.points[1],
                          (200,200,200), thickness)
            
        return self.points, image3
    

def crop_image(image, **kwargs):
    '''Function to crop images using mouse
    
    [image2, points] = crop_image(image, **show, **cmap, **window_name)
    
    image: input image.
    show: 'True' to show the image with rectangle to be cropped. Otherwise, the
          image will not be shown.
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name.
    image2: output imaga, with the cropping rectangle drown in it.
    points: a variable of type 'list' with the 2 main points to draw the crop-
            ping rectangle (upper-left and lower-right).
    
    How to use: 
        1. Left mouse button - choose the rectangle corners to crop the image
        (upper-left and lower-right). If two points were chosen, the rectangle
        will be completed and the function end.
        2. Right mouse click - erase the chosen points and starts the choose
        again from begening.
    '''
    # Discovering the image type [color (img_type = 3) or gray (img_type = 2)]
    img_type = len(np.shape(image))
    
    # Obtaining '**kwargs'
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # If there is no a window name chose, apply the standard one.
    if window_name is None:
        window_name = "Choose a region to Crop"
    # Calling class:
    cropping_class = crop_image1()
    [points, image3] = cropping_class.run(image, cmap, window_name, img_type)
    
    # If 'show = True', show the final image, with the chosen cropping rectang.
    if show == True:
        window_name2 = 'Crop result'
        cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name2,image3)
    
    # Cropping image
    points1 = np.asarray(points[0]) # 'points' is a 'list' variable, so we used
    points2 = np.asarray(points[1]) # np.asarray to obtain an array of values.
    # Forming a vector with 4 points (x1, y1, x2, y2).
    points3 = np.concatenate((points1, points2))
    
    # Using rectng. to really crop the image. This 'IF' logic is stated because
    # we need to know the corners that were chosen, and its sequence.
    if points3[2] - points3[0] > 0:         # if x2 > x1 in (x1, y1),(x2,y2)
        if points3[3] - points3[1] > 0:     # if y2 > y1 in (x1, y1),(x2,y2)
            image2 = image[points3[1]:points3[3], points3[0]:points3[2]]
        else:                               # if y1 > y2 in (x1, y1),(x2,y2)
            image2 = image[points3[3]:points3[1], points3[0]:points3[2]]
    else:
        if points3[3] - points3[1] > 0:
            image2 = image[points3[1]:points3[3], points3[2]:points3[0]]
        else:
            image2 = image[points3[3]:points3[1], points3[2]:points3[0]]
    
    return image2, points



class crop_multiple1(object):
    
    def __init__(self):
        self.window_name = 'Choose the area to crop'    # Our window's name.
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.center = []        # Center point to crop the image.

    # This function defines what happens when a mouse event take place.    
    def mouse_actions(self, event, x, y , buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # If we have a mouse move, update the current position. We need this to
        # draw a 'working rectangle' when the user moves mouse (to show the
        # possible rectangle that the user could choose).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate that the user chose the center point.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.center.append((x, y))
    
    # This function actually run the 'crop_multiple' function
    def run(self, image, width, height, img_type, cmap):
        if cmap is not None:
            if img_type == 2:
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'self.window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(self.window_name, self.mouse_actions)
        
        # Loop to draw the 'working rect.' while the user choose the final one.
        while (not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            cv2.imshow(self.window_name, image2)
            
            # The further logic implement a sum or a subtraction to width/2 and
            # height/2 from the central to obtain the rectangle corners. Exem-
            # ple: The upper-left corner is obtained subtracting width/2 and
            # height/2 from the central point coordinates.
            
            # We make 'width % 2' and 'height % 2' because 'number % 2' disco-
            # ver if its number is even or odd. And if one of this is odd, we
            # need to add 'one' when summing width/2 or height/2 to the center
            # point coordinates (otherwise the first chosen rectangle will not
            # macht with the other ones).
            if width % 2 == 0:
                if height % 2 == 0:
                    difference1 = tuple((int(np.int(width/2)), int(np.int(height/2))))
                    difference2 = tuple((int(-np.int(width/2)), int(-np.int(height/2))))
                else:
                    difference1 = tuple((int(np.int(width/2)), int(np.int(height/2)+1)))
                    difference2 = tuple((int(-np.int(width/2)), int(-np.int(height/2))))
            else:
                if height % 2 == 0:
                    difference1 = tuple((int(np.int(width/2)+1), int(np.int(height/2))))
                    difference2 = tuple((int(-np.int(width/2)), int(-np.int(height/2))))
                else:
                    difference1 = tuple((int(np.int(width/2)+1), int(np.int(height/2)+1)))
                    difference2 = tuple((int(-np.int(width/2)), int(-np.int(height/2))))
            
            # To find the rectang. corners we subtract 'width/2' and '-width/2'
            # from the 'central point' (or 'self.current', where the mouse is).
            point1 = np.subtract(self.current,difference1)
            point2 = np.subtract(self.current,difference2)
            point1 = tuple(point1)      # It was more easy to use tuple in the
            point2 = tuple(point2)      # function 'cv2.rectangle'.
            # Defining thickness based on image size
            thickness = int(np.shape(image2)[1]/500)
            cv2.rectangle(image2, point1, point2, (200,200,200), thickness)
            cv2.imshow(self.window_name, image2)
            
            # If a center point was already chosen (center > 0) or if the 'ESC'
            # key was pressed (k = 27), exit the 'while loop'.
            center = np.asarray(self.center)
            k = cv2.waitKey(50) & 0xFF
            if center.any() > 0 or k == 27:
                self.done = True
        
        # Using rect. to really crop the image. This 'IF' logic is stated be-
        # cause we need to know the corners that were chosen, and its sequence.
        if point2[0] - point1[0] > 0:         # if x2 > x1 in (x1, y1),(x2,y2)
            if point2[1] - point1[1] > 0:     # if y2 > y1 in (x1, y1),(x2,y2)
                image3 = image[point1[1]:point2[1], point1[0]:point2[0]]
            else:                             # if y1 > y2 in (x1, y1),(x2,y2)
                image3 = image[point2[1]:point1[1], point1[0]:point2[0]]
        else:
            if point2[1] - point1[1] > 0:
                image3 = image[point1[1]:point2[1], point2[0]:point1[0]]
            else:
                image3 = image[point2[1]:point1[1], point2[0]:point1[0]]
        
        # Closing this window we prevent that this window remain after running
        # this functino.
        cv2.destroyWindow(self.window_name)
        return image3, point1, point2



def crop_multiple(images, cmap):
    '''Function to crop multiple images with the same rectangle.
    
    [images2, points] = crop_multiple(images, cmap)
    
    images: input image (a'list' or a 'numpy.ndarray' variable). The I size has
            to be: 'I.shape = (n, heigh, width)', with 'n = number of images'.
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    images2: output cropped images (a 'numpy.ndarray' variable)
    points: a variable of type 'list' with the 2 main points to draw the crop-
            ping rectangle (upper-left and lower-right).

    First image:
        1. Left mouse button - choose the rectangle corners to crop the image
        (upper-left and lower-right). If two points were chosen, the rectangle
        will be completed and the function end.
        2. Right mouse click - erase the chosen points and starts the choose
        again from begening.
        
    Onother images:
        1. Move mouse to select where the center of cropping rectangle will be.
    
    This function uses mouse to choose a rectangle to crop in the first image,
    and uses the mouse again to place this same rectangle (same in dimentions)
    to crop the other images in a different place (different in terms of (x,y))
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)
    
    # Discovering the image type [color (img_type = 4) or gray (img_type = 3)]
    img_type = len(np.shape(images))
    
    # First image cropping uses the 'crop_image' function.
    [I00, points0] = crop_image(I[0,...], cmap=cmap)
    
    if img_type == 3:       # Grayscale images
        I2 = np.zeros((len(I), I00.shape[0], I00.shape[1]), np.uint8)
        I2[0,...] = I00
    elif img_type == 4:     # Color images
        I2 = np.zeros((len(I), I00.shape[0], I00.shape[1], I00.shape[2]), np.uint8)
        I2[0,...] = I00
    
    # Here we create 'lists' to put the points that are the rectangle corners.
    pointA = []
    
    pointA.append((min(points0[0][0],points0[1][0]), min(points0[0][1],points0[1][1])))
    
    # Taking 'points' from a 'list' variable to a 'numpy array' one.
    points1 = np.asarray(points0[0])
    points2 = np.asarray(points0[1])
    points3 = np.concatenate((points1, points2))    # concatenation points.
    
    # With the points information, we can obtain the width and height
    width = abs(points3[2] - points3[0])
    height = abs(points3[3] - points3[1])
    
    # For loop to perform a crop in all the images in 'I'
    for n in range(1,len(I)):
        # The best practice is every time call the class before use its functi.
        crop_class = crop_multiple1()
        [I2[n,...], point1, point2] = crop_class.run(I[n], width, height,
                                                     img_type, cmap)
        pointA.append((min(point1[0],point2[0]), min(point1[1],point2[1])))
    
    return I2, pointA



class crop_poly_multiple1(object):
    
    def __init__(self):
        self.window_name = 'Choose the area to crop'    # Our window's name.
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.center = []        # Center point to crop the image.

    # This function defines what happens when a mouse event take place.    
    def mouse_actions(self, event, x, y , buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # If we have a mouse move, update the current position. We need this to
        # draw a 'working polygon' when the user moves mouse (to show the
        # possible polygon that the user could choose).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate that the user chose the center point.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.center.append((x, y))
    
    # This function actually run the 'crop_multiple' function
    def run(self, image, points, equalize, img_type, cmap, show, window_name):
        self.window_name = window_name
        if equalize == True:
            image = cv2.equalizeHist(image)
        if cmap is not None:
            if img_type == 'gray':
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'self.window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(self.window_name, self.mouse_actions)
        
        # Loop to draw the 'working polygon' while user choose the final one.
        points1 = np.array(points,dtype='int32')
        while (not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            cv2.imshow(self.window_name, image2)
            
            # We have to sum the polygon position with the current mouse one:
            current1 = np.array(self.current,dtype='int32')
            
            # The next few lines is to read the user keyboard. We needed to de
            # fine something global ('global key1') to take this value outside
            # the class.
            def on_press(key):
                global key1
                key1 = key
                return False
            
            def on_release(key):
                global key1
                key1 = 'None'
                return False
            # To turn on the listener who read the keyboard
            listener = keyboard.Listener(on_press=on_press,
                                         on_release=on_release)
            listener.start() 
            time.sleep(0.05) # waiting few miliseconds.
            
            # This is to bring points1 to the center where mouse is.
            points1[:,0] = points1[:,0] - np.mean(points1[:,0]) + current1[0]
            points1[:,1] = points1[:,1] - np.mean(points1[:,1]) + current1[1]
            
            # If user use 'up' and 'down' arrows, the ROI will rotate more
            # then if the user press 'left' or 'right' arros.
            try:
                if str(key1) == 'Key.up':
                    points1 = rotate2D(points1,current1,ang=+np.pi/16)
                elif str(key1) == 'Key.down':
                    points1 = rotate2D(points1,current1,ang=-np.pi/16)
                elif str(key1) == 'Key.left':
                    points1 = rotate2D(points1,current1,ang=+np.pi/64)
                elif str(key1) == 'Key.right':
                    points1 = rotate2D(points1,current1,ang=-np.pi/64)
                else:
                    pass
            except: pass
            
            # To use 'polylines', we have to enter with a 'int32' array with
            # a specific shape, then we use 'reshape'
            points1 = np.array(points1,np.int32)
            # This is to make the shape aceptable in 'cv2.polylines' and 
            # 'cv2.fillConvexPoly'
            points1 = points1.reshape((-1, 1, 2))
            
            # Defining line thicknesse based on image size
            if np.shape(image2)[0] > 600:
                thickness = int(np.shape(image2)[0]/300)
            else:
                thickness = 2
            
            cv2.polylines(image2,[points1],True,(200,200,200), thickness)
            cv2.imshow(self.window_name, image2)
            points1 = np.squeeze(points1, axis=1)
            # If a center point was already chosen (center > 0) or if the 'ESC'
            # key was pressed (k = 27), exit the 'while loop'.
            center = np.asarray(self.center)
            k = cv2.waitKey(50) & 0xFF
            if center.any() > 0 or k == 27:
                self.done = True
        
        points1 = points1.reshape((-1, 1, 2))
        image3 = np.zeros(np.shape(image),np.uint8)
        cv2.fillConvexPoly(image3, np.array(points1,dtype='int32'),
                           (255,255,255))
        # Closing this window we prevent that this window remain after running
        # this function (if show is not True).
        if show is not True:
            cv2.destroyWindow(self.window_name)
        # Since we change the shape of 'points1', we need to use np.squeeze.
        return image3, np.squeeze(points1, axis=1)



def crop_poly_multiple(images, **kwargs):
    '''Function to crop multiple images with the same polygon.
    
    [images2, points] = crop_poly_multiple(images, cmap)
    
    images: input image (a'list' or a 'numpy.ndarray' variable). The I size has
            to be: 'I.shape = (n, heigh, width)', with 'n = number of images'.
    cmap: Chose the prefered colormap. Use 'None' for color or grayscale image.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    equalize: When choosen 'True', the showed image has histogram equalization.
    show: When choosen 'True', the final image with polyroi is showed.
    window_name: You can choose the window name.
    images2: output cropped images (a 'numpy.ndarray' variable)
    points: a 'list' variable with the first point selected in each polygon.

    First image:
        1. Left mouse button - to choose the polygon corners.
        2. Right mouse click - to finish.
        
    Onother images:
        1. Move mouse to select where the center of cropping rectangle will be.
    
    This function uses mouse to choose a polygon to crop in the first image,
    and uses the mouse again to place this same polygon (same in dimentions)
    to crop the other images in a different place (different in terms of (x,y))
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)

    # Obtaining '**kwargs'
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    equalize = kwargs.get('equalize')
    
    # If there is no a window name chose, apply the standard one.
    if window_name is None:
        window_name = "Choose a region to Crop"
    
    # Discovering the image type [color (img_type1 = 3), gray (img_type1 = 4)]
    img_type1 = len(np.shape(images))
    if img_type1 == 3:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # First image cropping uses the 'polyroi' function.
    I2 = []
    if equalize == True:
        I[0,...] = cv2.equalizeHist(I[0,...])
    [Itemp, points] = polyroi(I[0,...], cmap = cmap, window_name = window_name)
    I2.append(Itemp)
    # First saved point.
    pointA = []
    pointA.append(np.asarray(points,np.int32))
    
    # For loop to perform a crop in all the images in 'I'
    for n in range(1,len(I)):
        # The best practice is every time call the class before use its functi.
        crop_class = crop_poly_multiple1()
        [Itemp, points1] = crop_class.run(I[n], points, equalize,
                                          img_type, cmap, show, window_name)
        I2.append(Itemp)
        pointA.append(points1)
    
    return I2, pointA



def filter_finder(x, y, **kwargs):
    '''
    Function to study which filter to use for a signal f(x) = y

    Input Parameters
    ----------------
    x : 1D numpy.ndarray
        Variable of f(x) function. It can be, for example, the time in a time-
        dependent f(x).
    y : 1D numpy.ndarray
        Signal or sequence to be filtered.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    
    
    How to Use the Function
    -----------------------
    
    output = filter_finder(x, y, **kwargs)
    
    inputs:
    show (string): operation mode, to choose what the function will plot. If
    'time' is chosen, this function compare the original signal with filtered
    one in the time domain (or in whatherver domain the original signal is). If
    'frequency' is chosen, the signal and filter spectra are compared. If
    'interpolation' is chosen, the original signal is compared with the inter-
    polated one for visualization. Standard value is 'time'.
    N (int): the number of points to be used in FFT and 'x' interpolation.
    fs (int): frequency of sampling
    Wn (int): critical frequency
    order (int): filter order
    
    output:
    output (tuple): Depends on the mode. If the mode is 'frequency', return w
    (wavelengths) and h (frequency response, to be ploted e.g. as np.abs(h)).
    If the mode is 'time', it returns the x and the y for the filtered signal.
    
        
    If show = 'frequency' compare signal and filter in the frequency domain.
    If show = 'time' compare signal and filtered signal in the time domain.
    If show = 'interpolation', it compares the true signal with its interpola-
    tion (to see if interpolation is good for this signal).
    '''
    N = kwargs.get('N')
    fs = kwargs.get('fs')
    Wn = kwargs.get('Wn')
    order = kwargs.get('order')
    show = kwargs.get('show')
    window = kwargs.get('window')
    
    if N is None:
        N = 2**10
    if fs is None:
        fs = 1
    if Wn is None:
        Wn = 0.25
    if order is None:
        order = 5
    if window is None:
        window = signal.windows.hamming(N)
    if show is None:
        show = 'time'
    # Defining 'T' as 1 over the sampling frequency 'fs'
    T = 1/fs
    # First interpolating the signal
    f = interp1d(x, y)
    x_ip = np.linspace(x[0], x[-1], N, endpoint=True)
    y_ip = f(x_ip)
    if show == 'interpolation':
        plt.subplots()
        plt.plot(x, y)
        plt.plot(x_ip, y_ip, 'o')
    # Calculating the FFT of the signal y(x)
    y_w = y_ip*window
    yf = fft(y_w)
    yf_log = 20*np.log10(np.abs(yf[0:N//2])/np.max(np.abs(yf[0:N//2])))
    xf = fftfreq(N, T)[:N//2]
    # Plotting the freq. response of the signal y(x)
    if show == 'frequency':
        plt.subplots()
        plt.plot(xf, yf_log)
    # Calculating the IIR filter for this signal
    sos = signal.iirfilter(order, Wn=Wn, fs=fs, btype='low', ftype='butter',
                           output='sos')
    w, h = signal.sosfreqz(sos, worN=N, fs=fs)
    if show == 'frequency':
        plt.plot(w, 20*np.log10(np.abs(h)), label='Filter')
        plt.show()
        return (w, h)
    if show == 'time':
        y_filt = signal.sosfiltfilt(sos, y)
        plt.subplots()
        plt.plot(x, y, label='Original signal')
        plt.plot(x, y_filt, label='Filtered signal')
        plt.legend()
        plt.show()
        return (x, y_filt)



def highpass_gauss(img, **kwargs):
    '''
    Apply a Gaussian-based high-pass filter
    
    I = highpass_gauss(img, **kwargs)
    
    'img' (np.array): image to be filtered (2D np.array)
    'sigma' (float): sigma from Gaussian function to be used in the filtering
    'bkg' (int): integer to be used as a background
    return (np.array): filtered image, rescaled to the 0-255 range.
    
    This functino apply a Gaussian-based high-pass filter by subtracting a
    gaussian filtered image from the original one (resulting in the high-freq.
    content). This function is very useful, and the first choise for testing
    high-pass filters in a broad range of samples.    
    '''
    ksize = kwargs.get('ksize')
    sigma = kwargs.get('sigma')
    bkg = kwargs.get('bkg')
    if ksize is None:
        ksize = (11,11)
    if sigma is None:
        sigma = 15
    if bkg is None:
        bkg = 127
    
    return scale255(img - cv2.GaussianBlur(img, ksize, sigma))



def highpass_fft(img, **kwargs):
    '''
    Apply a high-pass FFT filter in a 2D image
    
    I = highpass_fft(img, **kwargs)
    
    'image' (np.array): image to be filtered (2D np.array)
    'frac' (float): fraction from the image to be removed from the low-frequen-
    cy part of the spectrum (standard: 0.25) (if passed a value greater than
    0.5, the function will assign 'frac = 0.5').
    'prnt' (boolean): true to print the images that yield from processing for
    testing purpose (standard: False).
    'title' (string): test to be added as a title in the printed images if
    'prnt = True'
    'return' (np.array): return the final image, after FFT filtering, rescaled
    to the 0-255 range.
    
    This function passes a high-pass filter based on eliminating a fraction of
    the FFT image spectrum. The fraction to be eliminated is stated in 'frac'
    and accounts for half of the spectrum. so if you choose 'frac=0.5' the
    whole spectrum will be eliminated, and no image will return. Note that the
    returned image is rescaled to the 0-255 range.
    
    **kwargs indicates the inputs that can be passed to the function, but are
    not mandatory.    
    '''
    frac = kwargs.get('frac')       # Fraction to be removed from spectrum
    prnt = kwargs.get('prnt')       # To print the images from the processing
    title = kwargs.get('title')     # To add a title to the printed images
    if frac is None:
        frac = 0.25
    elif frac > 0.5:
        frac = 0.5
    if prnt is None:
            prnt = False
    if title is None:
        title = None
    # First we calculate the bidimensional FFT of input image
    im_fft = fftpack.fft2(img)
    
    if prnt:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(np.log(np.abs(im_fft)))
        ax[0].axis('off')
        plt.suptitle(title)
    # Here we remove the parts from low frequencies, by vanishing its content
    r, c = im_fft.shape
    im_fft[0:int(frac*r),:] = 0
    im_fft[r-int(frac*r):r,:] = 0
    im_fft[:,0:int(frac*c)] = 0
    im_fft[:,c-int(frac*c):c] = 0
    
    if prnt:
        ax[1].imshow(np.log(np.abs(im_fft)))
        ax[1].axis('off')
    # Then we calculate the inverse FFT of transformed image
    final = fftpack.ifft2(im_fft).real
    
    if prnt:
        ax[2].imshow(final)
        ax[2].axis('off')
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.01,
                            right=0.98, hspace=0.01, wspace=0.01)
    
    return scale255(final)



def lowpass_fft(img, **kwargs):
    '''
    Apply a low-pass FFT filter in a 2D image
    
    I = lowpass_fft(img, **kwargs)
    
    'image' (np.array): image to be filtered (2D np.array)
    'frac' (float): fraction from the image to be removed from the high-
    frequency part of the spectrum (standard: 0.25) (if passed a value greater
    than 0.5, the function will assign 'frac = 0.5').
    'prnt' (boolean): true to print the images that yield from processing for
    testing purpose (standard: False).
    'title' (string): test to be added as a title in the printed images if
    'prnt = True'
    'return' (np.array): return the final image, after FFT filtering, rescaled
    to the 0-255 range.
    
    This function passes a low-pass filter based on eliminating a fraction of
    the FFT image spectrum. The fraction to be eliminated is stated in 'frac'
    and accounts for half of the spectrum. so if you choose 'frac=0.5' the
    whole spectrum will be eliminated, and no image will return. Note that the
    returned image is rescaled to the 0-255 range.
    
    **kwargs indicates the inputs that can be passed to the function, but are
    not mandatory. 
    '''
    frac = kwargs.get('frac')       # Fraction to be removed from spectrum
    prnt = kwargs.get('prnt')       # To print the images from the processing
    title = kwargs.get('title')     # To add a title to the printed images
    if frac is None:
        frac = 0.25
    elif frac > 0.5:
        frac = 0.5
    if prnt is None:
            prnt = False
    if title is None:
        title = None
    # First we calculate the bidimensional FFT of input image
    im_fft = fftpack.fft2(img)
    
    if prnt:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(np.log(np.abs(im_fft)))
        ax[0].axis('off')
        plt.suptitle(title)
    # Here we devide by 'frac' to use 'frac' as a fraction o which will be re-
    # moved
    r, c = im_fft.shape
    im_fft[int(0.5*r-frac*r):int(0.5*r+frac*r),:] = 0
    im_fft[:,int(0.5*c-frac*c):int(0.5*c+frac*c)] = 0
    
    if prnt:
        ax[1].imshow(np.log(np.abs(im_fft)))
        ax[1].axis('off')
    # Then we calculate the inverse FFT of transformed image
    final = fftpack.ifft2(im_fft).real
    
    if prnt:
        ax[2].imshow(final)
        ax[2].axis('off')
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.01,
                            right=0.98, hspace=0.01, wspace=0.01)
    
    return scale255(final)



def filt_hist(hist):
    '''
    Functino to 'filter' equalized histogram, removing the zeros values 
    between positive ones (usefull to process equalized historgrams).
    
    output = filt_hist(hist)
    
    input:
    hist (np.array): input histogram to be 'filtered' (removed the zero values)
    
    output:
    output (np.array): output histogram without zero values between positive
    ones.
    '''
    output = np.zeros(np.shape(hist))
    output[0] = hist[0]
    # We go until the middle of the signal (len(hist)//2)
    active = False
    for n in range(1, len(hist)//2):
        if (hist[n-1]>0) and (hist[n]==0):
            output[n] = hist[n-1]
            active = True
        if hist[n]>0:
            active = False
        if active:
            output[n] = output[n-1]
        else:
            output[n] = hist[n]
    # Then we come back from the end to the middle, to do not make the final
    # zero values from the spectrum becoming non-zero values by this function.
    output[-1] = hist[-1]
    active = False
    for n in range(len(hist)-2, len(hist)//2-1, -1):
        if (hist[n+1]>0) and (hist[n]==0):
            output[n] = hist[n+1]
            active = True
        if hist[n]>0:
            active = False
        if active:
            output[n] = output[n+1]
        else:
            output[n] = hist[n]
    return output



def imroiprop(I):
    """Function to get properties of a ROI in a image
    
    [props, Imask] = imroiprop(I)
    
    props[0]: sum all pixel values in ROI;
    props[1]: mean of non-zero values in ROI;
    props[2]: std of non-zero values in ROI.
    """
    # First we choose a polygonal ROI:
    [Imask, points] = polyroi(I, cmap = cv2.COLORMAP_PINK)
    
    # The mask poits of ROI came with "255" value, but we need the value "1".
    Imask[Imask > 0] = 1
    
    # Preparing a vector to receive variables:
    props = np.zeros(3, 'float')
    
    # Multiplying by mask
    Itemp = I*Imask
    # Integrating all the pixel values:
    props[0] = np.sum(Itemp)
    # Mean pixel value from ROI:
    props[1] = Itemp[Itemp!=0].mean()
    # Standar deviation from pixels in ROI:
    props[2] = Itemp[Itemp!=0].std()
    
    return props, Imask



def roi_stats(experiments, colors, **kwargs):
    ''' Easely calculate statistics of images in a given region of the images
    
    This function uses a interactive graphical user interface (GUI) to calcula-
    te the statistics of multiple images, in a given region of interest (ROI)
    inside the image. To use this function, your images have to be in a folder
    tree like bellow. The outer folder will be 'Images Folder', that has to be
    passed to this function in the 'images_dir' variable. Inside this folder
    you can add as many parts of your experiment you want to (in this case
    exemplified as animals). Inside each part (or animal) of your experiments,
    it have to be folders corresponding to the exact experiments conducted (for
    example different times, or differents treatments types or measurements).
    After the processing, a '.csv' data file is saved in the folder/directory
    specified in the variable 'save_dir'.
    
    Images Folder
        |
        |
        |--- Animal 1
        |       |
        |       |--- Experiment 1
        |       |
        |       |--- Experiment 2
        |       |
        |       '--- Experiment 3
        |
        |
        '--- Animal 2
                |
                |--- Experiment 1
                |
                |--- Experiment 2
                |
                '--- Experiment 3
    
    Parameters
    ----------
    experiments : list
        Names of the experiments. Note that it has to mach the experiment names
        e.g. ['Experiment 1', 'Experiment 2', 'Experiment 3'] in the example.
    colors : list
        Name of the colors (or channels) to be analized in the image, or a list
        with the string 'gray' for grayscale images. E.g.: ['gray'] or ['red'],
        or even all the colors ['red', 'green', 'blue']
    
    **kwargs (arguments that may or may not be passed to the function)
    ----------
    images_dir : string
        Root directory (outer folder) of your images. E.g. 'C:/Users/User/data'
    save_dir : string
        Directory to save the images.
    stats : list
        A list with the statistics to be calculated. Only mean and standard
        deviation are suported until now. E.g. ['mean'] or ['mean', 'std']
    show : boolean
        If 'True', print all the images processed. The default is 'False'.
    colormap : int
        The colormap to use while choosing the region of interest
        examples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV, cv2.COLORMAP_PARULA.
    '''
    colormap = kwargs.get('colormap', cv2.COLORMAP_PINK)
    stats = kwargs.get('stats', ['mean', 'std'])
    images_dir = kwargs.get('images_dir', None)
    save_dir = kwargs.get('save_dir', None)
    show = kwargs.get('show', False)
    
    ## Display a message to choose the folder the folders
    # Create the root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    ## Prompt a dialog to user to select the folder 'images_dir' and 'save_dir'
    if not images_dir:
        images_dir = filedialog.askdirectory(title='Please select the folder where the images are')
        if not images_dir:
            messagebox.showinfo('Selection Cancelled', 'No folder selected for images.')
    
    # Prompt the user to select the second folder
    if not save_dir:
        save_dir = filedialog.askdirectory(title='Please select the folder to save files')
        if not save_dir:
            messagebox.showinfo('Selection Cancelled', 'No folder selected to save the data.')
    
    # Show a message confirming the selections
    messagebox.showinfo('Folders Selected', f'Images folder: {images_dir}\nSave folder: {save_dir}')
    
    # Verify if user choose the colors to be processed
    if not colors or not experiments:
        raise ValueError('\n\n- You did not choose the colors or the experiments correctly')

    # Entering into the folder of images
    folders = list_folders(images_dir)
    
    # Initiating the variable with the data to be saved
    dados = {'Experiment':[]}
    
    # Adding all the names of experiments to be saved on 'dados'
    for exp in experiments:
        for color in colors:
            for stat in stats:
                dados[exp+' - '+stat+' '+color] = []
    
    # Defining the question to be asked to the user
    question = 'OK: to continue\n\nCancel: for redo'
    
    
    # Iterating between the folders with images
    for folder in folders:
        # Adding the experiment's name in the data to be saved
        dados['Experiment'].append(folder)
        # Reading all experiment names inside 'folders'
        path = os.path.join(images_dir, folder)
        times = list_folders(path)
        # Iteratinf through all the experiments
        for exp in experiments:
            if exp in times:
                # Reading the name of the first image inside the experiment folder
                path = os.path.join(images_dir, folder, exp)
                name = list_images(path)[0]
                path = os.path.join(path, name)
                # Reading image
                if 'gray' in colors:
                    Itemp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                else:
                    Itemp = cv2.imread(path, cv2.IMREAD_COLOR)
                    # Trying to change the image colors, if it is not, the image
                    # was probably not recognized correctly, so raise an error
                    try:
                        Itemp = cv2.cvtColor(Itemp, cv2.COLOR_BGR2RGB)
                    except:
                        raise ValueError(f'\n\n- Error when opening image from {folder}, for the experiment: {exp}')
                asw = 2
                while asw != 1:
                    # With the next command the user can circulate the lesion
                    [Imask, points] = polyroi(Itemp, cmap = colormap,
                                                  window_name = f'Select the region of interest for {folder} / {exp}')
                    # Then the box dialog pop up itself:
                    asw = ctypes.windll.user32.MessageBoxW(0,question,'Question', 1)
                
                # Printing the image, if 'show = True'
                if show:
                    plt.subplots()
                    plt.imshow(Itemp)
                    points.append(points[0])
                    points = np.asarray(points)
                    plt.plot(points[:, 0], points[:, 1])
                    plt.tight_layout()
                    plt.show()
                
                # Calculating the measures defined in 'stast'
                if 'gray' in colors:
                    if 'mean' in stats:
                        dados[f'{exp} - mean gray'].append(np.mean(Itemp[Imask>0]))
                    if 'std' in stats:
                        dados[f'{exp} - std gray'].append(np.std(Itemp[Imask>0]))
                else:
                    if 'red' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean red'].append(np.mean(Itemp[:,:,0][Imask[:,:,0]>0]))
                        if 'std' in stats:
                            dados[f'{exp} - std red'].append(np.std(Itemp[:,:,0][Imask[:,:,0]>0]))
                    if 'green' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean green'].append(np.mean(Itemp[:,:,1][Imask[:,:,1]>0]))
                        if 'std' in stats:
                            dados[f'{exp} - std green'].append(np.std(Itemp[:,:,1][Imask[:,:,1]>0]))
                    if 'blue' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean blue'].append(np.mean(Itemp[:,:,2][Imask[:,:,2]>0]))
                        if 'std' in stats:
                            dados[f'{exp} - std blue'].append(np.std(Itemp[:,:,2][Imask[:,:,2]>0]))
            
            # If does not find the data, add zero to the data to be saved
            else:
                if 'gray' in colors:
                    if 'mean' in stats:
                        dados[f'{exp} - mean gray'].append(int(0))
                    if 'std' in stats:
                        dados[f'{exp} - std gray'].append(int(0))
                else:
                    if 'red' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean red'].append(int(0))
                        if 'std' in stats:
                            dados[f'{exp} - std red'].append(int(0))
                    if 'green' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean green'].append(int(0))
                        if 'std' in stats:
                            dados[f'{exp} - std green'].append(int(0))
                    if 'blue' in colors:
                        if 'mean' in stats:
                            dados[f'{exp} - mean blue'].append(int(0))
                        if 'std' in stats:
                            dados[f'{exp} - std blue'].append(int(0))
    
    ## Saving the data
    columns = ['Experiment']
    for exp in experiments:
        for color in colors:
            for stat in stats:
                columns.append(f'{exp} - {stat} {color}')
    
    # Go to the directory where the data will be saved
    os.chdir(save_dir)
    
    # Monting the data to be saved on a DataFrame
    df = pd.DataFrame(dados, columns = columns)
    
    # Actually saving the data
    df.to_csv('dados.csv', index = False)



def imchoose(images, cmap):
    '''Function to chose images of given image set.
    
    chosen = imchoose(images, cmap)
    
    images: input images (a'list' or a 'numpy.ndarray' variable). The I shape
            has to be: 'np.shape(I)=(n, heigh, width)', with 'n' being the
            number of images.
    cmap: Chose the prefered pyplot colormap (it is a 'string' variable). Use
          'None' for color or grayscale image. Some colormap exemples: pink,
          CMRmap, gray, RdGy, viridis, terrain, hsv, jet, etc.
    chosen: A 'int' column vector with '1' for a chosen image, and '0'
            otherwise, in the column position corresponpding the image
            position. OBS: 'len(chosen) = len(images)'.

    How it works:
        Click in the image number that you want to choose. The image will
        chang, and will appear '(chosen)'. After choose all images, press
        'enter' or 'esc' key.
    OBS: this function create a object named 'key1'!
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)
    
    done = False    
    chosen = np.squeeze(np.zeros([1,len(I)]),axis=0)
    # Calling figure before printing the images
    fig, ax = plt.subplots(1,len(I))
    for n in range(0,len(I)):
        ax[n].imshow(I[n], cmap)
        ax[n].axis('off')
    # Next 'adjust'  to show the images closer.
    plt.subplots_adjust(top=0.976,bottom=0.024,left=0.015,right=0.985,
                        hspace=0.0,wspace=0.046)
    while(not done):
        # The next few lines is to read the user keyboard. We needed to de
        # fine something global ('global key1') to take this value outside
        # the class 'Listener'.
        def on_press(key):
            global key1
            key1 = key
            # When we recurn 'False', the listener stops, we always stops be-
            # cause otherwise the key always is the pressed key (looks like the
            # user is pressing the key continously, what affects the logic).
            return False
        
        def on_release(key):
            return False
        # To turn on the listener who read the keyboard
        listener = keyboard.Listener(on_press=on_press,
                                     on_release=on_release)
        listener.start() 
        # The 'plt.pause' allow the 'pyplot' have time to show the image.
        plt.pause(2) # waiting few miliseconds.
        # If user press 'esc' or 'enter', the program closes
        try:
            # If user press 'enter' or 'esc', the program ends.
            if str(key1) == 'Key.esc':
                done = True
            elif str(key1) == 'Key.enter':
                done = True
            # The user will chose pressing a key number (1,2,3,etc). This has
            # to add the value '1' (that means image chosen) to 'chosen' in the
            # position 'int(temp)-1', because remember Python starts in '0'.
            if str(key1) is not None and len(str(key1)) < 7:
                temp = np.mat(str(key1))
                chosen[int(temp)-1] = 1
                
                if chosen.any() > 0:
                    # Plotting the image
                    for n in range(0,len(I)):
                        if chosen[n] == 1:
                            # We change I[n] to become a little white:
                            Itemp = np.uint8(255*((I[n]+50)/np.max(I[n]+50)))
                            ax[n].imshow(Itemp, cmap)
                            ax[n].axis('off')
                            ax[n].set_title('(Escolhida)')
                        else:
                            ax[n].imshow(I[n], cmap)
                            ax[n].axis('off')
                    plt.subplots_adjust(top=0.976,bottom=0.024,left=0.015,
                                        right=0.985,hspace=0.0,wspace=0.046)
                    plt.pause(2)
        except: pass
    plt.close(fig)
    return np.array(chosen, 'int')



def align_features(I1, I2, draw):
    ''' Align images with Feature-Based algorithm, from OpenCV
    
    [Ir, warp] = align_features(I1, I2, draw)
    
    Parameters
    ----------
    I1 : numerical-array (grayscale)
        Image to be aligned (array).
    I2 : numerical-array (grayscale)
        Reference image.
    draw: if 'True', an image with the chosen features is ploted.

    Returns
    -------
    Ir : numpy-array
        Aligned image.
    warp : 3x3 numpy-array
        Warping matrix.

    '''
    MAX_FEATURES = 500       # Make  **kwargs
    GOOD_MACH_PERCENT = 0.20 # Make  **kwargs
    
    # Detect ORB features and compute descriptors:
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(I1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(I2, None)
    
    # Match features:
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score:
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Removing some matches:
    numGoodMatches = int(len(matches)*GOOD_MACH_PERCENT)
    matches = matches[:numGoodMatches] # choose the good ones
    # matches = matches[len(matches)-numGoodMatches:] # choose not so good ones
    
    # Drawing the classified matches:
    if draw == True:
        imMatches = cv2.drawMatches(I1,keypoints1, I2,keypoints2, matches,None)
        plt.subplots()
        plt.imshow(imMatches)
        plt.axis('off')
    
    # Extracting location of good matches:
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for n, match in enumerate(matches):
        points1[n, :] = keypoints1[match.queryIdx].pt
        points2[n, :] = keypoints2[match.trainIdx].pt
    
    # Finding homography that align this two images
    warp, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Using homography
    height, width = I1.shape[0], I1.shape[1]
    Ir = cv2.warpPerspective(I1, warp, (width, height))
    
    return Ir, warp



def align_ECC(images, warp_mode):
    '''Thi function align a set of gray imagess using a function from 'OpenCV'
    
    [Ia, warp_matrix] = align_ECC(images, warp mode)
    
    images: these are the input images, with all the images that we want to
            align, respecting the standard images shape '(n, widt, height)',
            where 'n' is the image number (n.max() = number of images).
    
    warp_mode: the transformation needed in the form: cv2.MOTION_TRANSF, subs
               tituing 'TRANSF' by: TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY.
    
    Ia: aligned images with shape = (n, height, width), being n = number of
        images.
    
    '''
    # Chossing the kind of image motion (image warp, or warp mode)
    if warp_mode == None:
        warp_mode = cv2.MOTION_TRANSLATION
    
    # Transforming in numpy array (from a possible 'tuple' or 'list')
    images1 = images.copy()
    images2 = np.asarray(images1)
    
    # Creating the warp matrix, the matrix with transformation coefficients, to
    # trnsform a missalined image in a aligned one (or to perform ANY transfor-
    # mation in a given image/matrix. At first, it will be a identity matrix.
    # In the HOMOGRAPHY mode, we need a 3x3 matrix, otherwise, a 2x3.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix0 = np.eye(3, 3, dtype = np.float32)
    else:
        warp_matrix0 = np.eye(2, 3, dtype = np.float32)
                    
    n_iterations = 500     # Maximum number of iterations.
    epsilon = 1e-5 # Threshold increment of correlation coef. betwn iterations
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations,
                epsilon)
    
    # To allocate a 'warp_matrix1' for every image, in a new array called warp_
    # matrix1, of shape = (n, warp_matrix1(x), warp_matrix1(y)), with n = number
    # of images. P.S.: warp_matrix1[0] is not used.
    warp_shape = warp_matrix0.shape
    warp_matrix1 = np.zeros((len(images2),warp_shape[0],warp_shape[1]))
       
    # This for is to find the transformation that make one image into
    # a aligned one (a 3x3 matrix). In this case, the algorithm used
    # is ECC, described in the paper (DOI: 10.1109/TPAMI.2008.113).
    for n in range(1,len(images2)):
        (cc, warp_matrix1[n,...]) = cv2.findTransformECC(images2[n,...],
                                                         images2[0,...],
                                                         warp_matrix0,
                                                         warp_mode, criteria)
#                                                         inputMask = None,
#                                                         gaussFiltSize = 5)
    
    # We will need the size of 'I' images.
    size = images2.shape[1:]
    
    # Loop to apply transformations chosen (with ECC) for all images.
    Ia = np.zeros((len(images2), size[0], size[1]), np.uint8)
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    for n in range(1,len(images2)):
        warp_matrix2 = warp_matrix1[n,...]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            Ia[n,...] = cv2.warpPerspective(images2[n,...], warp_matrix2,
                                             (size[1], size[0]), flags)
        else:
            Ia[n,...] = cv2.warpAffine(images2[n,...], warp_matrix2,
                                        (size[1], size[0]), flags)
    
    # The first aligned image is the original one (the reference image).
    Ia[0,...] = images2[0,...]
    
    # Defining the output warp matrix (will be = warp_matrix1, but without the
    # first empty part (warp_matrix1[0,:,:], that is always zero))
    warp_matrix = warp_matrix1[1:,:,:]
                
    return Ia, warp_matrix



def imwarp(images, warp_matrix):
    '''This function warp (using cv2.warpAffine) a given image (or a set of
    images) using the input warp matrix:
        
    images_output = imwarp(images, warp_matrix)
    
    OBS: This 'warp_matrix' is iqual to that 'warp_matrix' output from 
    # 'align_ECC' function.
    '''
    # To assing a value to variable 'size', we need to assing zero in it.
    size = np.zeros(2, np.uint64)
    
    numb = []
    # Discovering the size of images. If len(np.shape(images)) = 2, we have
    # only one image to warp, if it is = 3, we  have a set of images to warp.
    if len(np.shape(images)) == 2:
        size[0] = np.shape(images)[0]
        size[1] = np.shape(images)[1]
        numb = 1                        # number of images to warp.
    elif len(np.shape(images)) == 3:
        size[0] = np.shape(images)[1]
        size[1] = np.shape(images)[2]
        numb = np.shape(images)[0]      # number of images to warp.
        
    # This flag is necessario to warp an image.
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    
    if numb == 1:
        # Defining the size/shape of the output image (or images).
        images_output = np.zeros((size[0], size[1]), np.uint8)
        warp_matrix2 = warp_matrix[0,...]
        images_output = cv2.warpAffine(images, warp_matrix2,
                                       (size[1], size[0]), flags)
    else:
        images_output = np.zeros((numb, size[0], size[1]), np.uint8)
        # This loop is to warp all the images we have.
        for n in range(0, numb+1):
            warp_matrix2 = warp_matrix[n,...]
            images_output = cv2.warpAffine(images[n], warp_matrix2,
                                           (size[1], size[0]), flags)
    
    return images_output



def roi_stats_in_detph(folder, numb, **kwargs):
    '''This function loads all the images inside a folder, define a region of
    interest inside each image (using two lines), devide this region in various
    equally spaced areas (isoareas), and calculates statistics for the pixels'
    intensity in each of these area, following a particular direction (going
    from one of the lines defined to the other one).
    
    Applications:
        Calculate fluorescence inside a tumor, as a function of depth, in his-
        tological slides.
        Microscopy, medical imaging, or material science, application that cal-
        culates pixels' statistics for different position in a given direction.
    
    Usage example:
        # Choose the folder where the images are
        folder = r'C:/Users/user/data'
        
        # Choose the number of isoareas to calculate
        numb = 10
        
        # Call the function, defining the channels to enter in the statistics
        dictionary = roi_stats_in_detph(folder, numb, channels=[1, 2, 3])
    
    Detailed explanation:
        
    1. Loading: This function will enter the folder difined in the variable
    'folder', as in the above example, and process all the images inside it.
    
    2. Choosing Lines: Then you will choose two lines (the front line and the
    back one), using a graphical user interface (GUI). These lines will define
    the region where the statistics will be calculated.
    
    3. Isoareas: The closed region defined by the two lines drawn by the user
    will be separated into various equally spaced lines (isolines). The area
    defined between two adjascent 'isolines' will be called an 'isoarea'.
    
    4. The Mask: An additional mask will be choosen by the user. Only the
    pixels inside this mask will be processed. Use this mask if you wants to
    select just part of the region of interest defined by the two lines drawn
    (to process just part of the isoareas). Otherwise, choose the intire region
    of interest to process all the selected pixels (all isoareas).
    
    5. Statistics in a Particular Direction: After that, a detailed statistics
    will be calculated for each isoarea (mean, standard deviation, mode, median
    and entropy), following a particular direction: going from the front line
    to the back line. The number of isoareas is defined in 'numb'.
    
    
    Input Parameters
    ----------------
    folder : string
        The directory where the images you want to process are.
    
    numb : integer
        The number of isoareas you want to calculate and process.
    
    Optional Parameters (kwargs)
    ----------------------------
    channels : list
        List here all the channels to be processed, e.g. 'channels = [1, 2, 3]'
        to process all the three channels of an image. Default value is '[1]'.
        In the case of grayscale images, you can use 'channels = [1]'.
    
    pixel_size : float or integer (default = 1.0)
        Enter the physical size discribed by a pixel. For example, if each
        pixel represents a size of 0.83 micrometers in the image (for a micro-
        scope image), than choose 'pixel_size = 0.83e-6'. Default value is 1.0.
    
    show : boolean
        Choose 'True' to visualize each image processed, with its isoareas.
    
    Returns (Outputs)
    -----------------
    dictionary : dictionary
        This function returns a dictionary with the statistics calculated for
        each channel selected in the variable 'channels'.'''
    
    # Obtaining the channels the user wants to process
    channels = kwargs.get('channels', [1])
    channels = np.asarray(channels)
    
    # Obtaining the physical image size of each pixel
    pixel_size = kwargs.get('pixel_size', 1)
    
    # Verifying if it is a test to print some images and some data
    test = kwargs.get('test')
    
    # If 'show = True', print each processed image for user's visualizagion
    show = kwargs.get('show')
    
    # Reading image names
    image_names = list_images(folder)
    
    # Then we start a loop to read and to process all images in 'folder'.
    for m, name in enumerate(image_names):
        # Reading images, whether in '.lsm' or not (may not prepared for gray lsm)
        if ('.lsm' in name) or ('.LSM' in name):
            I = read_lsm(os.path.join(folder, name))
        else:
            I = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        
        # Verifying which channel will be used, and saving it to the 'I1' variable
        if (1 in channels) and (len(channels)==1):
            I1 = I[:, :, 0]
        elif (2 in channels) and (len(channels)==1):
            I1 = I[:, :, 1]
        elif (3 in channels) and (len(channels)==1):
            I1 = I[:, :, 2]
        else:
            # First, we extract the 'm' image from the folder.
            I1 = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        
        # Choosing the ROI of the first function.
        window = 'Choose the front line'
        [Itemp, points1] = polyroi(I1, cmap = cv2.COLORMAP_PINK,
                                   window_name = window)
        points1 = np.asarray(points1)  # 'polylines' function require in array.
        
        # Drawing the chosen points in the begining image.
        I2 = I1.copy()
        cv2.polylines(I2, [points1], False, (220,200,200), 3)
        
        # Choosing the ROI of the second function.
        window = 'Choose the back line'
        [Itemp, points2] = polyroi(I2, cmap = cv2.COLORMAP_PINK,
                                   window_name = window)
        # 'polylines' needs points as an array.
        points2 = np.asarray(points2)
        
        # If the user choose 'points1' in one direction and 'points2' in another
        # one, we need to reverse the 'points2' direction to the code works
        if np.linalg.norm(points1[0, :]-points2[0, :]) > np.linalg.norm(points1[0, :]-points2[-1]):
            points2 = points2[::-1]
        
        # The first and the last points of the first line will be used in 2nd line
        points2[0,:] = points1[0,:]
        points2[-1,:] = points1[-1,:]
        
        # Printing testing image, if in testing is activated
        if test:
            cv2.polylines(I2, [points2], False, (220,200,200), 3)
            plt.subplots()
            plt.title('I2 with points2')
            plt.imshow(I2)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Next few lines will interpolate the points with the same number of pixels
        # of the image height. In order to make the isolines, it is very important
        # that 'points1' and 'points2' have the same number of points
        points1a = np.zeros([int(abs(points1[-1,0]-points1[0,0])),2], 'float')
        points2a = np.zeros([int(abs(points1[-1,0]-points1[0,0])),2], 'float')
        
        tck1, u1 = splprep([points1[:,0], points1[:,1]], s=0)
        tck2, u2 = splprep([points2[:,0], points2[:,1]], s=0)
        
        points1a[:,0], points1a[:,1] = splev(np.linspace(0, 1, len(points1a[:,1])), tck1)
        points2a[:,0], points2a[:,1] = splev(np.linspace(0, 1, len(points1a[:,1])), tck2)
        
        # Printing some images if testing is activated
        if test:
            plt.subplots()
            plt.plot(points1a[:,0], points1a[:,1])
            plt.plot(points2a[:,0], points2a[:,1])
            plt.plot(points1[:,0], points1[:,1], '.')
            plt.plot(points2[:,0], points2[:,1], '.')
            plt.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(I2)
            ax[0].axis('off')
            Itemp = I1.copy()
            Itemp = cv2.polylines(Itemp, [points1a.astype('int')], False, (220,200,200), 3)
            Itemp = cv2.polylines(Itemp, [points2a.astype('int')], False, (220,200,200), 3)
            Itemp = cv2.polylines(Itemp, [points1.astype('int')], False, (220,200,200), 3)
            Itemp = cv2.polylines(Itemp, [points2.astype('int')], False, (220,200,200), 3)
            ax[1].imshow(Itemp)
            ax[1].axis('off')
            plt.tight_layout()
            plt.show()
        
        # Joining all isolines in just one variable (better to handle), using the
        # lines chosen above (left and right lines). It has a size of '2*numb+2'
        # because each line is defined by 2 columns, and we need one lines more to
        # define 'numb' isoareas (e.g. 2 isoareas needs 3 lines to be defined).
        isolines = np.zeros([len(points1a[:,0]), 2*numb+2], 'float')
        diff = (points2a - points1a)/numb
        isolines[:,0:2] = points1a[:,0:2]
        for n in range(numb+1):
            isolines[:, n*2] = points1a[:, 0] + n*diff[:, 0]
            isolines[:, n*2+1] = points1a[:, 1] + n*diff[:, 1]
        
        # If testing, print the isolines
        if test:
            plt.subplots()
            plt.title('isolines = points1a and points2a')
            for n in range(0, numb+1):
                plt.plot(isolines[:, 2*n], isolines[:, 2*n+1])
            plt.tight_layout()
            plt.show()
        
        # The next steps create the images with the ROIs of isolines. 'I4' is
        # only to user see, and 'I5' is what we'll use.
        I4 = []
        I5 = []
        # Defining the variable that will help us to "paint" the image 'I4'.
        factor = np.round(250/numb)
        # Using 'For' loops to create the images I4 (to see) and I5 (to calculate)
        Itemp1 = np.zeros(np.shape(I1), 'uint8')
        for n in range(0, numb):
            # We use a pair of isolines to create a closed isoarea with 'fillPoly'
            pts_temp = np.concatenate((isolines[:,0+n*2:2+n*2],
                                       isolines[::-1,2+n*2:4+n*2]))
            pts_temp = np.matrix.round(pts_temp)
            pts_temp = np.array(pts_temp, 'int')
            cv2.fillPoly(Itemp1, [pts_temp],
                         (factor*(n+1),factor*(n+1),factor*(n+1)))
        # I4 and Itemp1 are just for user view, next are the ones to really calcul.
        for n in range(0, numb):
            Itemp2 = np.zeros(np.shape(I1), 'uint8')
            pts_temp = np.concatenate((isolines[:,0+n*2:2+n*2],
                                       isolines[::-1,2+n*2:4+n*2]))
            pts_temp = np.matrix.round(pts_temp)
            pts_temp = np.array(pts_temp, 'int')
            cv2.fillPoly(Itemp2, [pts_temp], (1,1,1))
            I5.append(Itemp2)
        
        # Transforming 'I4' and 'I5' from list to 'numpy array'.
        I4 = I1.copy()
        I4 = scale255(I4)
        I4 = 0.65*I4 + 0.35*Itemp1
        I4 = np.matrix.round(I4)
        I4 = np.array(I4, 'uint8')
        I5 = np.asarray(I5)
        
        # Printing the final image for user's verification, if 'show = True'
        if show:
            plt.subplots()
            plt.imshow(I4, cmap='RdGy')
            plt.axis('off')
            plt.title('Plotting I4 image')
            plt.tight_layout()
            plt.show()
        
        # If testing, printing one image more
        if test:
            fig, ax = plt.subplots(1,numb)
            fig.suptitle('I5')
            for n in range(0,numb):
                ax[n].imshow(I5[n], cmap='RdGy')
                ax[n].axis('off')
            plt.tight_layout()
            plt.show()
        
        # Chosing the region where fluorescence will be calculated (fluorescen-
        # ce will be calculated inside this region ('Imask'), in each isoarea).
        window = 'Choose a mask where the pixels will be evaluated'
        [Imask, points3] = polyroi(I4, cmap = cv2.COLORMAP_PINK,
                                             window_name = window)
        # We need '1' and '0' pixels values to be the ROI that will multiply
        # our isoareas.
        Imask[Imask > 0] = 1
        
        ## The mean width between the isoareas will be estimated using two mag-
        ## nitudes: the mean vectorial distance between isoareas ('vector1' in
        ## next lines), and the vectorial distance pointing the depth (which is
        ## perpendicular to 'vector2' in next lines). Width will be the part of
        ## 'vector1' that is perpendicular to 'vector2'
        
        # First, let us find the center of mass of all ROIs in image I5 using scipy
        centers = []
        for image in I5:
            center = center_of_mass(image)
            centers.append(center)
        
        # Printing the centers of mass inside the image for testing
        if test:
            plt.subplots()
            plt.title('Center of Mass for Each ROI')
            plt.imshow(I4, cmap='RdGy')
            for center in centers:
                plt.plot(int(center[1]), int(center[0]), marker='.', markersize=10)
            plt.axis('off')
            plt.tight_layout()
            plt.show
        
        # Finding the mean vectorial distance between centroids. This will give us
        # the mean distance between isoareas and its direction (used in the future)
        vectors = []
        last = centers[0]
        # Finding distance between each pair of adjascent center/isoarea
        for center in centers[1:]:
            vectors.append(np.array(center)-np.array(last))
            last = center
        vectors = np.asarray(vectors)
        # Averaging distances to find final mean distance between isoareas/centers
        vector1 = np.mean(vectors, axis=0)
        
        # Here, the depth direction will be estimated by the difference between the
        # first and the last point of the isolines. Note that the points are
        # defined as (x axis, y axis) in Python, which is different from images,
        # which are defined as (line, column). That is the because we changed
        # "x" with "y" with using the indexing "[::-1]"
        vector2 = isolines[0:1,0:2] - isolines[len(isolines)-1::,0:2]
        vector2 = vector2[0][::-1]
        
        ## Finding the sine between 'vector1' and 'vector2' to calculate width
        # First, calculating the cross product
        cross_product = np.cross(vector1, vector2)
        
        # Then calculating the magnitudes
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)
        
        # Calculating the sine
        sin_angle = cross_product / (mag1 * mag2)
        sin_angle = np.clip(sin_angle, -1, 1)
        # Calculating angle in radians and transforming to degrees
        angle = np.arcsin(sin_angle)
        
        # Mean width can be estimated by finding the part of 'vector1' that is
        # in the direction of the depth (or perpendic. to 'vector2'), this
        # multiplied by the 'pixel_size'
        width = abs(np.sin(angle)*mag1*pixel_size)
        
        # Creating a dictionary to save the data (based on the channels choosen)
        dictionary = {'width': []}
        for channel in channels:
            dictionary[f'mean of CH{channel}'] = []
            dictionary[f'std of CH{channel}'] = []
            dictionary[f'mode of CH{channel}'] = []
            dictionary[f'median of CH{channel}'] = []
            dictionary[f'entropy of CH{channel}'] = []
        
        # Preparing data to be saved on 'name.csv'
        for n in range(0, numb):
            dictionary['width'].append(n*width + width/2)
            for channel in channels:
                # Since Python starts with '0', we subtracts '1' from the channel
                Itemp = I[:, :, channel-1]*I5[n]*Imask
                dictionary[f'mean of CH{channel}'].append(Itemp[Itemp!=0].mean())
                dictionary[f'std of CH{channel}'].append(Itemp[Itemp!=0].std())
                dictionary[f'mode of CH{channel}'].append(stats.mode(Itemp[Itemp!=0], axis = None)[0])
                dictionary[f'median of CH{channel}'].append(np.median(Itemp[Itemp!=0]))
                dictionary[f'entropy of CH{channel}'].append(shannon_entropy(Itemp[Itemp!=0]))
        
        # Entering folder 'results' and saving the data to 'name.csv'
        os.chdir(folder)
        path = 'results'
        if os.path.exists(path) is not True:
            os.mkdir(path)
        os.chdir(path)
        # Actually saving the data
        df = pd.DataFrame(dictionary, columns = list(dictionary.keys()))
        name = name.split('.')[0]
        df.to_csv(f'{name}.csv', index = False)
    
    # Returning the dictionary
    return dictionary



def good_colormaps(image):
    '''This function show a list of the good 'matplotlib.pyplot' colormaps:
    
    imfun.good_colormaps(image)
    
    Input Parameter
    ---------------
    image : numpy.ndarray
        Image to be printed in different colormaps. It should be a grayscale
        image in the format (H, W), where H is the higher and W is the image
        width.
    
    Some of the colormaps:
    prism
    terrain
    flag ** (a lot of contrast)
    pink *
    coolwarm
    nipy_spectral
    gist_stern **
    gist_ncar
    Spectral
    hsv
    jet
    CMRmap (article)
    viridis (parula) (article)
    gnuplot
    RdGy ***
    BrBG
    
    image = input image
    
    The program will output a sequence of three figures showing the colormaps
    '''
    name1 = "Normal published colormaps4"
    plt.figure(name1)
    plt.subplot(131).set_title('gray')
    plt.imshow(image, cmap = "gray")
    plt.xticks([]), plt.yticks([])
    plt.subplot(132).set_title('viridis')
    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap = "viridis")
    plt.subplot(133).set_title('CMRmap')
    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap = "CMRmap")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()  
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
    
    name2 = "Different Colormaps 14"
    plt.figure(name2)
    plt.subplot(231).set_title('prism')
    plt.imshow(image, cmap = "prism")
    plt.xticks([]), plt.yticks([])
    plt.subplot(232).set_title('terrain')
    plt.imshow(image, cmap = "terrain")
    plt.xticks([]), plt.yticks([])
    plt.subplot(233).set_title('flag')
    plt.imshow(image, cmap = "flag")
    plt.xticks([]), plt.yticks([])
    plt.subplot(234).set_title('gist_stern')
    plt.imshow(image, cmap = "gist_stern")
    plt.xticks([]), plt.yticks([])
    plt.subplot(235).set_title('pink')
    plt.imshow(image, cmap = "pink")
    plt.xticks([]), plt.yticks([])
    plt.subplot(236).set_title('nipy_spectral')
    plt.imshow(image, cmap = "nipy_spectral")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
    
    name3 = "Different Colormaps 24"
    plt.figure(name3)
    plt.subplot(231).set_title('gist_ncar')
    plt.imshow(image, cmap = "gist_ncar")
    plt.xticks([]), plt.yticks([])
    plt.subplot(232).set_title('RdGy')
    plt.imshow(image, cmap = "RdGy")
    plt.xticks([]), plt.yticks([])
    plt.subplot(233).set_title('gnuplot')
    plt.imshow(image, cmap = "gnuplot")
    plt.xticks([]), plt.yticks([])
    plt.subplot(234).set_title('gist_stern')
    plt.imshow(image, cmap = "gist_stern")
    plt.xticks([]), plt.yticks([])
    plt.subplot(235).set_title('hsv')
    plt.imshow(image, cmap = "hsv")
    plt.xticks([]), plt.yticks([])
    plt.subplot(236).set_title('BrBG')
    plt.imshow(image, cmap = "BrBG")
    plt.xticks([]), plt.yticks([])
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
