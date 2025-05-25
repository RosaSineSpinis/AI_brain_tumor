# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from _2_custom_datagen import imageLoader
from keras.models import load_model
import itertools
import glob

def plot_model_prediction(model, img, true_mask, axis='z', slice_indices=[64], class_count=4, show_channel=0):
    """
    Predicts and plots the input image, ground truth mask, and model prediction side-by-side.

    Parameters:
    - model: trained Keras model
    - img: 4D numpy array (H, W, D, C), input image volume
    - true_mask: 4D numpy array (H, W, D, classes), one-hot encoded ground truth mask
    - axis: str, one of 'x', 'y', 'z' (default='z')
    - slice_indices: list of ints, slice indices to plot along the chosen axis
    - class_count: int, number of classes (e.g., 4)
    - show_channel: int, which channel to display from the image (e.g., 0 for T1, FLAIR, etc.)
    """

    # Predict
    input_img = np.expand_dims(img, axis=0)
    pred_mask = model.predict(input_img, verbose=0)[0]
    pred_mask_classes = np.argmax(pred_mask, axis=-1)
    true_mask_classes = np.argmax(true_mask, axis=-1)

    # Plot
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(12, 4 * len(slice_indices)))

    if len(slice_indices) == 1:
        axes = np.expand_dims(axes, 0)

    for i, idx in enumerate(slice_indices):
        
        print("axis", axis)
        if axis == 'z':
            img_slice = img[:, :, idx, show_channel]
            true_slice = true_mask_classes[:, :, idx]
            pred_slice = pred_mask_classes[:, :, idx]
        elif axis == 'y':
            img_slice = img[:, idx, :, show_channel]
            true_slice = true_mask_classes[:, idx, :]
            pred_slice = pred_mask_classes[:, idx, :]
        elif axis == 'x':
            img_slice = img[idx, :, :, show_channel]
            true_slice = true_mask_classes[idx, :, :]
            pred_slice = pred_mask_classes[idx, :, :]
        else:
            raise ValueError("Axis must be one of 'x', 'y', 'z'.")

        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f"Input Slice ({axis}={idx})")

        axes[i, 1].imshow(true_slice, cmap='jet', vmin=0, vmax=class_count - 1)
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(pred_slice, cmap='jet', vmin=0, vmax=class_count - 1)
        axes[i, 2].set_title("Prediction")

        # for ax in axes[i]:
        #     ax.axis('off')

    plt.tight_layout()
    plt.show()



test_img_list = sorted(glob.glob("BraTS2020_TrainingData/input_data_128/train/images/image_*.npy"))
test_mask_list = sorted(glob.glob("BraTS2020_TrainingData/input_data_128/train/masks/mask_*.npy"))
print(test_img_list)
print()
print(test_mask_list)

my_model = load_model('brats_3d_LP_0.0005/brats_3d_epoch_40.hdf5', compile=False)

# for img_num in range(1,50):
for idx, img in enumerate(test_img_list):
    
    if idx == 11: break
    # img_num = 82
    # test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num)+".npy")
    # test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num)+".npy")
    test_img = np.load(test_img_list[idx])
    test_mask = np.load(test_mask_list[idx])

    test_mask_argmax=np.argmax(test_mask, axis=3)

    # test_img_input = np.expand_dims(test_img, axis=0)
    # test_prediction = my_model.predict(test_img_input)
    # test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]

    plot_model_prediction(my_model, test_img, test_mask, axis='z', slice_indices=[64, 70, 80])
    
