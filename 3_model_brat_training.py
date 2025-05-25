# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for 
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""


import os
import numpy as np
from _2_custom_datagen import imageLoader
#import tensorflow as tf
import keras
#import tensorflow.keras as keras
from matplotlib import pyplot as plt
import glob
import random
from keras.callbacks import CSVLogger, ModelCheckpoint



####################################################
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0,num_images-1)
test_img = np.load(train_img_dir+img_list[img_num])
test_mask = np.load(train_mask_dir+msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# %%


#############################################################
#Optional step of finding the distribution of each class and calculating appropriate weights
#Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

# import pandas as pd
# columns = ['0','1', '2', '3']
# df = pd.DataFrame(columns=columns)
# train_mask_list = sorted(glob.glob('BraTS2020_TrainingData/input_data_128/train/masks/*.npy'))
# for img in range(len(train_mask_list)):
#     print(img)
#     temp_image=np.load(train_mask_list[img])
#     temp_image = np.argmax(temp_image, axis=3)
#     val, counts = np.unique(temp_image, return_counts=True)
#     zipped = zip(columns, counts)
#     conts_dict = dict(zipped)
    
#     df = df.append(conts_dict, ignore_index=True)

# label_0 = df['0'].sum()
# label_1 = df['1'].sum()
# label_2 = df['1'].sum()
# label_3 = df['3'].sum()
# total_labels = label_0 + label_1 + label_2 + label_3
# n_classes = 4
# #Class weights claculation: n_samples / (n_classes * n_samples_for_class)
# wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
# wt1 = round((total_labels/(n_classes*label_1)), 2)
# wt2 = round((total_labels/(n_classes*label_2)), 2)
# wt3 = round((total_labels/(n_classes*label_3)), 2)

#Weights are: 0.26, 22.53, 22.53, 26.21
#wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
#These weihts can be used for Dice loss 
# %%


##############################################################
#Define the image generators for training and validation

train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))
##################################

########################################################################
batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
n_slice=70
plt.figure(figsize=(12, 8))

plt.subplot(331)
plt.imshow(img[0,:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(332)
plt.imshow(img[0,:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(333)
plt.imshow(img[0,:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(334)
plt.imshow(msk[0,:,:,n_slice,1])
plt.title('Mask1')
plt.subplot(335)
plt.imshow(msk[0,:,:,n_slice,2])
plt.title('Mask2')
plt.subplot(336)
plt.imshow(msk[0,:,:,n_slice,3])
plt.title('Mask3')
plt.show()

n_slice=70
plt.figure(figsize=(12, 8))
plt.subplot(331)
plt.imshow(img[1,:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(332)
plt.imshow(img[1,:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(333)
plt.imshow(img[1,:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(334)
plt.imshow(msk[1,:,:,n_slice,1])
plt.title('Mask1')
plt.subplot(335)
plt.imshow(msk[0,:,:,n_slice,2])
plt.title('Mask2')
plt.subplot(336)
plt.imshow(msk[0,:,:,n_slice,3])
plt.title('Mask3')
plt.show()
# %%


###########################################################################
#Define loss, metrics and optimizer to be used for training
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001 #0.01 #0.001 #0.0005 #0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################
#Fit the model 

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size
print("steps_per_epoch", steps_per_epoch)
print("val_steps_per_epoch", val_steps_per_epoch)

from  simple_3d_unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

import tensorflow as tf
print("tf ", tf.__version__)  # TensorFlow version
print("tf ", tf.keras.__file__)  # Location of tf.keras module

try:
    import keras
    print("keras ", keras.__version__)  # Standalone keras version
    print("keras ", keras.__file__)  # Location of the keras module
except ImportError:
    print("Standalone keras is not installed.")
    
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
"""
For test purposes only,
try to force overfitting with one sample
"""
model.summary()

# %%
"""
For test purposes only,
try to force overfitting with one sample
"""

# Get one batch of data from the generator
# x_batch, y_batch = train_img_datagen.__next__()  # Or next(train_img_datagen)
train_img_datagen = imageLoader(train_img_dir, ["image_115.npy", "image_116.npy"], 
                                train_mask_dir, ["mask_115.npy", "mask_116.npy"], batch_size)

x_batch, y_batch = train_img_datagen.__next__()

print(x_batch.shape)


# n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice=60
plt.subplot(331)
plt.imshow(x_batch[0,:,:,n_slice,0], cmap='gray')
plt.title('Image flair')
plt.subplot(332)
plt.imshow(x_batch[0,:,:,n_slice,1], cmap='gray')
plt.title('Image t1')
plt.subplot(333)
plt.imshow(x_batch[0,:,:,n_slice,2], cmap='gray')
plt.title('Image t1ce')
plt.subplot(334)
plt.imshow(y_batch[0,:,:,n_slice,0])
plt.title('Mask_1')
plt.subplot(335)
plt.imshow(y_batch[0,:,:,n_slice,1])
plt.title('Mask_2')
plt.subplot(336)
plt.imshow(y_batch[0,:,:,n_slice,2])
plt.title('Mask_2')
plt.subplot(337)
plt.imshow(y_batch[0,:,:,n_slice,3])
plt.title('Mask_3')
plt.show()

n_slice=60
plt.subplot(331)
plt.imshow(x_batch[1,:,:,n_slice,0], cmap='gray')
plt.title('Image flair')
plt.subplot(332)
plt.imshow(x_batch[1,:,:,n_slice,1], cmap='gray')
plt.title('Image t1')
plt.subplot(333)
plt.imshow(x_batch[1,:,:,n_slice,2], cmap='gray')
plt.title('Image t1ce')
plt.subplot(334)
plt.imshow(y_batch[1,:,:,n_slice,0])
plt.title('Mask_1')
plt.subplot(335)
plt.imshow(y_batch[1,:,:,n_slice,1])
plt.title('Mask_2')
plt.subplot(336)
plt.imshow(y_batch[1,:,:,n_slice,2])
plt.title('Mask_2')
plt.subplot(337)
plt.imshow(y_batch[1,:,:,n_slice,3])
plt.title('Mask_3')
plt.show()

############################################################
#%%
"""
For test purposes only,
try to force overfitting with one sample
check what is difference between number of pixels in background and object
"""
import numpy as np
labels = np.argmax(y_batch, axis=-1)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

#%%
"""
#%%
For test purposes only,
try to force overfitting with one sample
checj shape
"""
print("x_batch.shape ", x_batch.shape)
print("y_batch.shape ", y_batch.shape)
"""
For test purposes only,
try to force overfitting with one sample
check whether there is one hot encoding in the mask
"""
import numpy as np

def check_one_hot_3d(masks):
    # Value check: all values should be 0 or 1
    is_binary = np.all((masks == 0) | (masks == 1))
    # Sum across class axis should be 1 for each voxel
    sums_to_one = np.all(np.sum(masks, axis=-1) == 1)
    return is_binary and sums_to_one

print("One-hot encoding valid:", check_one_hot_3d(y_batch))

#%%
print("shape ", x_batch.shape)
print("shape[0] ", x_batch.shape[0])

#%%
directory = f'brats_3d_LP_nodrop_{LR}/'

if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory did not exist, so it was created.")
elif not os.listdir(directory):
    print("Directory exists but is empty.")
else:
    raise RuntimeError("Directory exists and is not empty.")
    
checkpoint = ModelCheckpoint(f'brats_3d_LP_nodrop_{LR}/'+'brats_3d_epoch_{epoch:02d}.hdf5', save_weights_only=False, save_freq='epoch')
csv_logger = CSVLogger(directory+'training_log.csv', append=True)
   
# Fit the model on just this batch
history = model.fit(
    x_batch,
    y_batch,
    batch_size=x_batch.shape[0],                # match the sample size
    epochs=150,
    validation_data=(x_batch, y_batch),  # overfit the same data
    callbacks=[csv_logger,checkpoint],
    verbose=1
)
#%%

# %%

csv_logger = CSVLogger('training_log.csv', append=True)

directory = f'brats_3d_LP_{LR}/'

if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory did not exist, so it was created.")
elif not os.listdir(directory):
    print("Directory exists but is empty.")
else:
    raise RuntimeError("Directory exists and is not empty.")
    
 
checkpoint = ModelCheckpoint(f'brats_3d_LP_{LR}/'+'brats_3d_epoch_{epoch:02d}.hdf5', save_weights_only=False, save_freq='epoch')


history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,    
          callbacks=[csv_logger, checkpoint]
          )

model.save('brats_3d.hdf5')
##################################################################

# %%

#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################
from keras.models import load_model

#Load model for prediction or continue training

#For continuing training....
#The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
#This is because the model does not save loss function and metrics. So to compile and 
#continue training we need to provide these as custom_objects.
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5')

#So let us add the loss as custom object... but the following throws another error...
#Unknown metric function: iou_score
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

#Now, let us add the iou_score function we used during our initial training
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

#Now all set to continue the training process. 
history2=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=1,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )
#################################################

#For predictions you do not need to compile the model, so ...
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
                      compile=False)

# %%

#Verify IoU on a batch of images from the test dataset
#Using built in keras function for IoU
#Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size=8 #Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#############################################
#Predict on a few test images, one at a time
#Try images: 
img_num = 82

test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num)+".npy")

test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


# print(test_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(test_prediction_argmax))


#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
plt.show()

############################################################

