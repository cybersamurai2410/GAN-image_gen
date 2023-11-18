from model import build_generator, build_discriminator

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)

        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    # Set up optimizer and loss function for both generator and discriminator
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)  # Compile the model with base class

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Retrieve images from batch and generate fake images using generator
        real_images = batch
        # fake_images = self.generator(tf.random.normal((128, 128)), training=False)

        batch_size = tf.shape(real_images)[0]
        fake_images = self.generator(tf.random.normal(shape=(batch_size, 128)), training=False)

        print("Batch Size:", batch_size)
        print("Real images shape:", real_images.shape)
        print("Fake images shape:", fake_images.shape)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real (0) and fake (1) images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add noise to the labels to improve discriminator performance
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate the discriminator loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Update discriminator weights via backpropagation
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate new images
            gen_images = self.generator(tf.random.normal((128, 128)), training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Update generator weights
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}


# Custom callback to monitor the model during training
class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))

        # Generate images from the random vectors
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        # Save generated images at the end of each epoch
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

# Load preprocessed dataset
load_directory = 'preprocessed_data'
image_shape = (128, 28, 28, 1)
dataset_structure = tf.TensorSpec(shape=image_shape, dtype=tf.float32)
ds = tf.data.experimental.load(load_directory, dataset_structure)

ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
ds = ds.batch(128, drop_remainder=True)
for batch in ds.take(1):
    print("Batch shape:", batch.shape)

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

generator = build_generator()
discriminator = build_discriminator()

fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

hist = fashgan.fit(ds, epochs=10, callbacks=[ModelMonitor()])

# generator.save('models/generatorx.h5')
# discriminator.save('models/discriminatorx.h5')

# Plot the training losses for generator and discriminator
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()
