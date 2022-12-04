import shutil
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Augment:
    Corrected_Path = None
    original_images_dir = None
    All_Data = None
    Parasitized_Data = None
    Uninfected_Data = None

    def Correctly_Move(self , path, org_dir):
        self.Corrected_Path = path
        self.original_images_dir = org_dir

        uninfected_path = os.path.join(org_dir, 'Uninfected')
        parasitized_path = os.path.join(org_dir, 'Parasitized')
        
        # moving uninfected images to the correct folder
        file = open(os.path.join(path,'False_parasitized.csv'), 'r')
        i = 0
        for line in file:
            if i == 0:
                i += 1
                continue
            else:
                line = line.split(',')
                line[1] = line[1].replace('\n', '')
                wrong_image_path = os.path.join(parasitized_path, line[1])
                # move to uninfected folder
                shutil.move(wrong_image_path, uninfected_path)
                # print(line[1] ,  'moved to uninfected folder')
                i+=1
        file.close()
        print(i , 'Done with uninfected images')
        # moving parasitized images to the correct folder
        file = open(os.path.join(path,'False_uninfected.csv'), 'r')
        i = 0
        for line in file:
            if i == 0:
                i += 1
                continue
            else:
                line = line.split(',')
                line[1] = line[1].replace('\n', '')
                wrong_image_path = os.path.join(uninfected_path, line[1])
                # move to parasitized folder
                shutil.move(wrong_image_path, parasitized_path)
                # print(line[1] ,  'moved to parasitized folder')
                i+=1
        file.close()   
        print(i , 'Done with parasitized images')
              
    def images_df(self, org_dir):
        parasitized_path = os.path.join(org_dir, 'Parasitized')
        Parasitized_Data_Frame = pd.DataFrame(columns=['image', 'image_path'])
        Uninfected_Data_Frame = pd.DataFrame(columns=['image', 'image_path'])
        All_Data_Frame = pd.DataFrame(columns=['image', 'image_path', 'label'])
        for files in os.listdir(parasitized_path):
            if files.endswith('.png'):
                # print('image' ,  files, 'image_path' , os.path.join(parasitized_path, files))
                Parasitized_Data_Frame.loc[len(Parasitized_Data_Frame)] = [files,  os.path.join(parasitized_path, files)]
                All_Data_Frame.loc[len(All_Data_Frame)] = [files, os.path.join(parasitized_path, files), 'Parasitized']
        for files in os.listdir(os.path.join(org_dir, 'Uninfected')):
            if files.endswith('.png'):
                # print('image' ,  files, 'image_path' , os.path.join(os.path.join(org_dir, 'Uninfected'), files))
                Uninfected_Data_Frame.loc[len(Uninfected_Data_Frame)] = [files,os.path.join(os.path.join(org_dir, 'Uninfected'), files)]
                All_Data_Frame.loc[len(All_Data_Frame)] = [files, os.path.join(os.path.join(org_dir, 'Uninfected'), files), 'Uninfected']

        self.All_Data = All_Data_Frame
        self.Parasitized_Data = Parasitized_Data_Frame
        self.Uninfected_Data = Uninfected_Data_Frame

        return Parasitized_Data_Frame, Uninfected_Data_Frame, All_Data_Frame

    def split_stratified_into_train_val_test(self , df_input, stratify_colname='y', frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None):
        if frac_train + frac_val + frac_test != 1.0:
            raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                            (frac_train, frac_val, frac_test))

        if stratify_colname not in df_input.columns:
            raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

        X = df_input # Contains all columns.
        y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                            y,
                                                            stratify=y,
                                                            test_size=(1.0 - frac_train),
                                                            random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                        y_temp,
                                                        stratify=y_temp,
                                                        test_size=relative_frac_test,
                                                        random_state=random_state)

        assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test
            
    def Augment_Data(self, df_train, df_val, df_test , output_dir):
        Data_type = ['train', 'val', 'test']
        Data_Lables = ['Parasitized', 'Uninfected']

        # created Augmented data folder if not exist
        os.mkdir(os.path.join(output_dir, 'Augmented_Data'))
        output_dir = os.path.join(output_dir, 'Augmented_Data')

        for data_type in Data_type:
            # making data type folder - Train, Val, Test
            os.mkdir(os.path.join(output_dir, data_type))
            for data_lable in Data_Lables:
                # creating lable folder - Parasitized, Uninfected inside Train, Val, Test . 
                # These folder will contain Augmented images of train, val, test.
                os.mkdir(os.path.join(output_dir, data_type, data_lable))
            if data_type == 'train':
                df = df_train.copy(deep = True)
            elif data_type == 'val':
                df = df_val.copy(deep = True)
            elif data_type == 'test':
                df = df_test.copy(deep = True)

            Augmentor =ImageDataGenerator(  rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='constant')
            
            groups=df.groupby('label') # group by class
            for label in df['label'].unique():  # for every class               
                group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
                sample_count=len(group)   # determine how many samples there are in this class  
                aug_img_count=0
                target_dir=os.path.join(output_dir,data_type, label)  # define where to write the images    
                aug_gen=Augmentor.flow_from_dataframe( group,  x_col='image_path', y_col=None, target_size=(128,128), class_mode=None,
                                                    batch_size=30, shuffle=False, save_to_dir=target_dir, save_prefix='aug-',
                                                    save_format='jpg')
                while aug_img_count<3*len(group):
                    images=next(aug_gen)            
                    aug_img_count += len(images) 