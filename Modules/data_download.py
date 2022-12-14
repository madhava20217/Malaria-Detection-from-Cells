import importlib.resources as pkg_resources
import os
import shutil
import opendatasets as od
import cv2
import glob
import pandas as pd

class Data_Download:
    '''Class for downloading and resizing data. Has utils for downloading data, resizing it and storing it in some specified location'''

    def __init__(self, data_dir):
        '''Initialiser for the Data_Download class
        
        Arguments:

        1. data_dir: the data directory as a relative path 
        
        
        Returns: a Data_Download object and downlads the dataset'''

        self.__url = "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria"
        self.data_path = data_dir

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if not os.path.exists(os.path.join(self.data_path, 'cell-images-for-detecting-malaria')):
            od.download(self.__url, self.data_path)
            shutil.rmtree(os.path.join(self.data_path, 'cell-images-for-detecting-malaria', 'cell_images', 'cell_images'))


    def resize_image(self, path, new_width, new_height, remove_misclassified = False):
        '''Function for resizing images and storing them in the specified path
        
        Arguments:
        
        1. path: where to store the resized image, path to the main directory
        2. new_width: new width after resizing
        3. new_height: new height after resizing
        4. remove_misclassified: whether to remove misclassified examples as per the study by Raihan et al. Experts found mislabeled examples in the dataset, which could be removed to not affect model training. The number of mislabeled examples are around 1400 in total, removal of which would still result in a large dataset.
        
        Returns: path to dataset directory (resizes images to the directory)'''
        data_dir = os.path.join(self.data_path, 'cell-images-for-detecting-malaria', 'cell_images')
        resized_dir = os.path.join(path, "Resized_data_{}_{}".format(new_width, new_height))

        if os.path.exists(resized_dir):
            shutil.rmtree(resized_dir)

        false_uninfected = pd.read_csv(os.path.join(os.path.dirname(__file__),'corrected_images','False_uninfected.csv'), index_col = 0)
        false_infected   = pd.read_csv(os.path.join(os.path.dirname(__file__),'corrected_images','False_parasitized.csv'), index_col = 0)

        false_uninfected = false_uninfected.False_uninfected.values
        false_infected   = false_infected.False_parasitized.values

        for dirpath, dir, files in os.walk(data_dir, topdown = True):
            if len(dir) != 0:
                for subdir in dir:
                    resized_dir_class  = os.path.join(resized_dir,subdir)

                    os.makedirs(resized_dir_class, exist_ok = True)
                    for file in glob.glob(os.path.join(data_dir, subdir, "*.png")):
                        fn = os.path.basename(file)
                        #checking for misclassified
                        if remove_misclassified:
                            if (fn in false_infected or fn in false_uninfected):
                                continue
                        img = cv2.imread(file)
                        img = img[...,::-1]
                        resized = cv2.resize(img, (new_width, new_height))
                        cv2.imwrite(os.path.join(resized_dir_class,
                                "{}".format(fn)), 
                            resized)
        
        print("Images resized to {} x {}".format(new_height, new_width))
        print("Directory path is {}".format(resized_dir))
        return resized_dir

    def remove_dataset(self):
        '''Removes the main dataset
        
        Arguments: None
        
        Returns: None'''
        shutil.rmtree(self.data_path)
        print("Data Directory removed")
