import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential #take in data in sequential manner
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D 
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import kaggle
from glob import glob
%matplotlib inline

base_skin_dir = os.path.join ( '..' , 'input' ) # changed it from base_dir to base_skin_dir becuase base_skin_dir was not defined until that

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))} 

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',

  
    'df': 'Dermatofibroma'
} 


skin_df = pd.read_csv(os.path.join(base_skin_dir, 'C:/Users/adity/HAM10000_metadata.csv'))
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

skin_def.head()

skin_df.isnull().sum()


skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
#Later on, may need to add null values for the other stuff, since the user may not input some of their info

skin_df.isnull().sum()


print(skin_df.dtypes)

fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

skin_df['localization'].value_counts().plot(kind='bar')

skin_df['age'].hist(bins=40)

skin_df['sex'].value_counts().plot(kind='bar')

sns.scatterplot('age','cell_type_idx',data=skin_df)

skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


