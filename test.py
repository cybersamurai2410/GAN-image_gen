from model import build_generator
import tensorflow as tf
import matplotlib.pyplot as plt
import os

generator = build_generator()
generator.load_weights(os.path.join('models', 'generatormodel.h5'))
imgs = generator.predict(tf.random.normal((16, 128, 1))) # Generating 16 images; The generator takes a random noise vector as input.

# Plot the generated images in a 4x4 grid
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4):
    for c in range(4):
        # Displaying the image at position [r, c] in the grid
        # The index for the image is calculated based on the current row and column
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1])
plt.show()
