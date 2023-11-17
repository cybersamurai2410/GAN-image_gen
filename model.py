import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization

'''Generator'''
def build_generator():
    model = Sequential()

    # Fully connected layer
    model.add(Dense(7 * 7 * 256, input_dim=128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 256)))

    # UpSampling and Convolutional Blocks
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

'''Discriminator'''
def build_discriminator():
    model = Sequential()

    # Convolutional Blocks
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Flatten then pass to dense layer to perform binary classification real/fake
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    '''Generate Samples'''
    generator = build_generator()
    generator.summary()

    # Generate 4 random noise samples and use the generator to create images
    img = generator.predict(np.random.randn(4, 128))
    print("Generated image shape:", img.shape)
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(img):
        ax[idx].imshow(np.squeeze(img)) # Plot the image using a specific subplot
        ax[idx].title.set_text(idx) # Appending the image label as the plot title
    plt.show()

    discriminator = build_discriminator()
    discriminator.summary()

    for idx, image in enumerate(img):
        # Reshape the image to add a batch dimension (1, height, width, channels)
        image_batch = np.expand_dims(image, axis=0)

        # Make a prediction with the discriminator
        prediction = discriminator.predict(image_batch)
        print(f"Discriminator's prediction on Sample {idx + 1}: {prediction[0][0]}")

        '''
        Discriminator's prediction on Sample 1: 0.5189899802207947
        Discriminator's prediction on Sample 2: 0.5189804434776306
        Discriminator's prediction on Sample 3: 0.5189709663391113
        Discriminator's prediction on Sample 4: 0.518961489200592
        '''
