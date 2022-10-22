import numpy as np
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Labelling:
    '''Class for labelling a dataset'''

    def label(self, directory, filename = 'labels.csv'):
        '''Function for labelling a dataset given a data directory
        
        Arguments:
        
        1. directory: The main directory containing the images
        2. filename: the filename to which the labels are to be saved
        
        Returns: None'''

        if not os.path.isdir(directory):
            raise Exception("Data Directory not found! Please run the data_download.ipynb notebook before proceeding.")
        file_list_parasitized = glob.glob(os.path.join(directory, 'Parasitized', '*.png'))
        file_list_uninfected  = glob.glob(os.path.join(directory, 'Uninfected', '*.png'))

        # length of filelists
        n_parasitized = len(file_list_parasitized)
        n_uninfected  = len(file_list_uninfected)

        file_list_parasitized = np.array(file_list_parasitized)
        file_list_uninfected  = np.array(file_list_uninfected)

        # to a numpy array
        file_list_parasitized = np.reshape(file_list_parasitized, newshape = (n_parasitized, 1))
        file_list_uninfected  = np.reshape(file_list_uninfected , newshape = (n_uninfected , 1))

        # labelling
        file_list_parasitized = np.append(file_list_parasitized, np.ones(file_list_parasitized.shape), axis = 1)
        file_list_uninfected  = np.append(file_list_uninfected , np.zeros(file_list_uninfected.shape), axis = 1)

        file_list = np.append(file_list_uninfected, file_list_parasitized, axis = 0)

        df = pd.DataFrame(file_list, columns = ['Image_Path', 'Parasitized'])

        df.to_csv(os.path.join(directory, filename), index = False)

    def train_test_val_split(self, 
                    directory, 
                    labels, 
                    train_split = 0.8, 
                    test_split = 0.1, 
                    stratify = True, 
                    random_state = 12345
                    ):
        '''Function to create training, testing and validation splits
        
        Arguments:
        
        1. directory: the root directory of the dataset
        2. label: the name of the CSV file containing labels
        3. train_split: ratio of samples for training
        4. test_split: ratio of samples for testing
        5. stratify: whether the splits should have the same distribution as the original dataset
        6. random_state: sets the seed for splitting
        Returns: None
        '''

        assert train_split > 0 and test_split >= 0
        assert train_split <= 1 and test_split <= 1
        assert train_split + test_split <= 1
        val_split_ratio = 1 - train_split - test_split
        df = pd.read_csv(os.path.join(directory, labels))
        train, test = train_test_split(df, 
                                    test_size = test_split, 
                                    stratify = df.Parasitized, 
                                    random_state = random_state)
        

        train_df = pd.DataFrame(train)
        train_df.reset_index(drop = True, inplace = True)

        test_df = pd.DataFrame(test)
        test_df.reset_index(drop = True, inplace = True)

        train, val = train_test_split(train_df, 
                                test_size = val_split_ratio/train_split,
                                stratify = train_df.Parasitized,
                                random_state = random_state)

        train_df = pd.DataFrame(train)
        train_df.reset_index(drop = True, inplace = True)

        val_df = pd.DataFrame(val)
        val_df.reset_index(drop = True, inplace = True)

        train_df.to_csv(os.path.join(directory, 'train.csv'), index = False)
        test_df.to_csv(os.path.join(directory, 'test.csv'), index = False)
        val_df.to_csv(os.path.join(directory, 'val.csv'), index = False)