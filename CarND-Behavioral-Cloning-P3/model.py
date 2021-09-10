# coding: utf-8

# In[1]:

##############################################################
#### Picking up Dataset provided by Udacity to train model ###
##############################################################

from urllib.request import urlretrieve
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile

def download_one(url, file):
    if not os.path.isfile(file):
        print("Download the file as it is... " + file + " ...")
        urlretrieve(url,file)
        print("the required file is downloaded")

download_one('https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip','data.zip') #s3 path of the dataset provided by udacity

print("All the required files are downloaded and is ready for extraction")

def uncompress_ft_lb(dir,name):
    if(os.path.isdir(name)):
        print('Data is  extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall('data')
uncompress_ft_lb('data.zip','data')

def data_Files(mypath):
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    print(onlyfiles)

print('All files downloaded and extracted as per the requirement')


# In[2]:

#Importing Required libraries to make the model work
import os
import csv


samples_space = [] #simple array to append all the entries present in the .csv file

with open('./data/data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples_space.append(line)


print("Done")


# In[3]:

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples_space,test_size=0.15) #simply splitting the dataset to train and validation set usking sklearn. .15 indicates 15% of the dataset is validation set


# In[4]:

import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#code for generator
def generator(samples_space, batch_size=32):
    num_samples = len(samples_space)
   
    while 1: 
        shuffle(samples_space) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples_space[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): 
                        
                        name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        center_angle = float(batch_sample[3]) 
                        images.append(center_image)
                        
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                    
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 
            


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



# In[7]:


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()


model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trim image 
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#layer 1-
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 2-
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 3-
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 4- 
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#layer 5- 
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#flatten image 
model.add(Flatten())

#layer 6- 
model.add(Dense(100))
model.add(Activation('elu'))

#Adding a dropout layer to avoid overfitting. 
model.add(Dropout(0.25))

#layer 7- 
model.add(Dense(50))
model.add(Activation('elu'))


#layer 8- 
model.add(Dense(10))
model.add(Activation('elu'))

#layer 9- 
model.add(Dense(1)) 


# the output is the steering angle
# using mean squared error loss function is the right choice for this regression problem
# adam optimizer is used here
model.compile(loss='mse',optimizer='adam')


#fit generator is used here as the number of images are generated by the generator
# no of epochs : 5

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

#saving model
model.save('model.h5')

print('Done! Model Saved!')

