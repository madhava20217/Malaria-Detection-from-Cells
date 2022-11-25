import cv2
import skimage.transform as transform
from math import radians
import random
import numpy as np

class Transformations():

    def __init__(self, image):
        self.image = image



    '''A class for transforming images to assist in data augmentation. Performs tasks like
    
    done - Rotate 
    - Shear
    - Translations
    done - Colour transformations
    done - Resizing
    done - Cropping black space 
    done - Random cropping
    done - Adding noise 
    done - Removing noise 
    
    '''
    
    def shear_transformation(self, image_arr, angle):
        '''Function to perform shear transformation
        
        Arguments:
        
        1. image_arr: the input image (as a numpy array)
        2. angle: the angle to be 'sheared' by in degrees 
        
        Returns a transformed image array'''
        return_image = transform.warp(image_arr, inverse_map= transform.AffineTransform(shear = radians(angle)))
        return return_image

    def translation_transformation(self, image_arr, x = 0, y = 0):
        '''Function to perform translation
        
        Arguments:
        
        1. image_arr: the input image (as a numpy array)
        2. x: the ratio to the width for the image to be horizontally shifted by
        3. y: the ratio to the height for the image to be vertically shifted by
        
        Returns a transformed image array'''
        height, width = image_arr.shape[0], image_arr.shape[1]
        transformation = transform.AffineTransform(translation=(x*width, y*height))
        return transform.warp(image_arr, inverse_map= transformation)

    def color_transformation(self, image_arr, kind = 'default'):
        '''Function for transforming a RGB image to some other space specified by the 'kind' argument.
        
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. kind : the transformation to be applied
        
        TODO: add types here
        
        Returns: The modified image (as a numpy array)
        
        **NOTE : HSV seems interesting**
        
        Consult here for transformations: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html'''

        
        if kind == 'default':
            return image_arr

        elif kind == 'gray':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)

        elif kind == 'bgr':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        
        elif kind == 'xyz':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2XYZ)
        
        elif kind == 'YCrCb':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2YCrCb)

        elif kind == 'HSV':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2HSV)

        elif kind == 'Lab':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2Lab)

        elif kind == 'Luv':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2Luv)

        elif kind == 'HLS':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2HLS)

        elif kind == 'YUV':
            return cv2.cvtColor(image_arr, cv2.COLOR_RGB2YUV)

        else:
            # default, do nothing
            return image_arr

    def rotate_transformation(self ,image_arr , angle):
        '''Function for rotating an image by some random angle. and then cropping the image to remove black borders.
        
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        
        Returns: The modified image (as a numpy array) '''
        num_rows, num_cols = image_arr.shape[:2]
        rotated_image = cv2.warpAffine(image_arr, cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, (num_cols, num_rows)))
        return rotated_image


    def crop_image(self, image_arr ,tolerance=0):
        '''Function for cropping an image by boundaries only.
        image is 2D or 3D image data
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. tolerance: the tolerance for cropping
        
        Returns: The modified image (as a numpy array) '''
        mask = image_arr>tolerance
        if image_arr.ndim==3:
            mask = mask.all(2)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return image_arr[row_start:row_end,col_start:col_end]

    def resize(self , image_arr , height , width):
        '''Function for resizing an image(2d/3d).
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. height: desired height of the image
        3. width: desired width of the image
        
        Returns: The modified image (as a numpy array) '''
        return cv2.resize(image_arr,(height,width))

    def random_crop(self , image_arr , crop_height , crop_width):
        '''Function for randomly cropping an image(2d/3d).

        Warning - the crop height and crop width should be less then the image height and width respectively.
        # handle this as assert madhava
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. crop_height: desired height of the image
        3. crop_width: desired width of the image
        
        Returns: The modified image (as a numpy array) '''
        
        height , width , _ = image_arr.shape
        max_x = image_arr.shape[1] - crop_width
        max_y = image_arr.shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image_arr[y: y + crop_height, x: x + crop_width]
        crop = cv2.resize(crop, (width, height))
        return crop

    def adding_noise(self, image_arr , kind='default' , std_devia = 1 , mean = 0 , noise_start = 0 , noise_end = 1):
        '''Function for resizing an image(2d/3d).
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. kind: the type of noise to be added
        3. std_devia: standard deviation of the noise (for gaussian kind only)
        4. mean: mean of the noise (for gaussian kind only)
        5. noise_start: the starting color value of the noise (for uniform and impulse kind only)
        6. noise_end: the ending color value of the noise (for uniform and impulse kind only)
        
        Returns: The modified image (as a numpy array) '''

        if kind == 'default':
            return image_arr

        elif kind == 'gaussian':
            num_height = 1 
            if(len(image_arr.shape) > 2):
                num_height = image_arr.shape[2]
            num_rows, num_cols = image_arr.shape[:2]
            gauss_noise=np.zeros((num_rows,num_cols , num_height),dtype=np.uint8)
            print(gauss_noise.shape) , print(image_arr.shape)
            cv2.randn(gauss_noise,std_devia,mean)
            gauss_noise=(gauss_noise*0.5).astype(np.uint8)
            gn_img=cv2.add(image_arr,gauss_noise)
            return gn_img

        elif kind == 'uniform':
            num_height = 1
            if(len(image_arr.shape) > 2):
                num_height = image_arr.shape[2]
            num_rows, num_cols = image_arr.shape[:2]
            uni_noise=np.zeros((num_rows,num_cols, num_height),dtype=np.uint8)
            cv2.randu(uni_noise,noise_start , noise_end)
            uni_noise=(uni_noise*0.5).astype(np.uint8)
            un_img=cv2.add(image_arr,uni_noise)
            return un_img
        
        elif kind == 'impulse':
            num_height = 1
            if(len(image_arr.shape) > 2):
                num_height = image_arr.shape[2]
            num_rows, num_cols = image_arr.shape[:2]
            imp_noise=np.zeros((num_rows,num_cols , num_height),dtype=np.uint8)
            cv2.randu(imp_noise,noise_start , noise_end)
            imp_noise=cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]
            in_img=cv2.add(image_arr,imp_noise)
            return in_img

        else:
            # default, do nothing
            return image_arr

    def removing_noise(self, image_arr , kind='default'):
        '''Function for denoising an image(2d/3d).
        Arguments: 
        
        1. image_arr: the input image (as a numpy array)
        2. kind: the kind of denoising you want to do
        
        Returns: The modified image (as a numpy array) '''
        if kind == 'default':
            return image_arr

        elif kind == 'gaussian':
            return cv2.GaussianBlur(image_arr,(3,3),0)

        elif kind == 'median':
            return cv2.medianBlur(image_arr,3)

        elif kind == 'bilateral':
            return cv2.bilateralFilter(image_arr,9,75,75)

        else:
            # default, do nothing
            return image_arr