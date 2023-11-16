import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization

load_directory = 'preprocessed_data'
image_shape = (128, 28, 28, 1)
dataset_structure = tf.TensorSpec(shape=image_shape, dtype=tf.float32)
ds = tf.data.experimental.load(load_directory, dataset_structure)

'''Generator'''
def build_generator():
    model = Sequential()

    # Fully connected layer
    model.add(Dense(7 * 7 * 256, input_dim=128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 256)))

    # Upsampling and Convolutional Blocks
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

    return model

generator = build_generator()
generator.summary()

# Generate 4 random noise samples and use the generator to create images
img = generator.predict(np.random.randn(4, 128))
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(img):
    ax[idx].imshow(np.squeeze(img)) # Plot the image using a specific subplot
    ax[idx].title.set_text(idx) # Appending the image label as the plot title
plt.show()
