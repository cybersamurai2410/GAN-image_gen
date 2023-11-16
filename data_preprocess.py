import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import os

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.config.list_physical_devices('GPU')
# tf.test.gpu_device_name()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ds = tfds.load('fashion_mnist', split='train')
data_iterator = ds.as_numpy_iterator()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample = data_iterator.next() # Grab an image and label
    ax[idx].imshow(np.squeeze(sample['image'])) # Plot the image using a specific subplot
    ax[idx].title.set_text(sample['label']) # Appending the image label as the plot title
plt.show()

# Scales the pixel values of the images from the range [0, 255] to [0, 1]. This scaling is done by dividing each pixel value by 255.
def scale_images(data):
    return data['image']/255

ds = ds.map(scale_images) # Running the dataset through the scale_images preprocessing step
ds = ds.cache() # Cache the dataset for that batch
ds = ds.shuffle(60000) # Shuffle data
ds = ds.batch(128) # Batch into 128 images per sample
ds = ds.prefetch(64) # While the model is training on the current batch, the next batch is already being loaded and prepared.

print(ds.as_numpy_iterator().next().shape)

save_directory = 'preprocessed_data'
tf.data.experimental.save(ds, save_directory)

