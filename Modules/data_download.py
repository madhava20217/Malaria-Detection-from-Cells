import os
import shutil
import opendatasets as od
import cv2
import glob

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


    def resize_image(self, path, new_width, new_height):
        '''Function for resizing images and storing them in the specified path
        
        Arguments:
        
        1. path: where to store the resized image, path to the main directory
        2. new_width: new width after resizing
        3. new_height: new height after resizing
        
        Returns: path to dataset directory (resizes images to the directory)'''
        data_dir = os.path.join(self.data_path, 'cell-images-for-detecting-malaria', 'cell_images')
        resized_dir = os.path.join(path, "Resized_data_{}_{}".format(new_width, new_height))

        for dirpath, dir, files in os.walk(data_dir, topdown = True):
            if len(dir) != 0:
                for subdir in dir:
                    resized_dir_class  = os.path.join(resized_dir,subdir)

                    os.makedirs(resized_dir_class, exist_ok = True)
                    for file in glob.glob(os.path.join(data_dir, subdir, "*.png")):
                        fn = os.path.basename(file)
                        img = cv2.imread(file)
                        resized = cv2.resize(img, (new_width, new_height))
                        cv2.imwrite(os.path.join(resized_dir_class,
                                "{}x{}{}".format(new_width, new_height, fn)), 
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
