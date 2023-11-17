from model import generator
import tensorflow as tf
import matplotlib.pyplot as plt
import os

generator.load_weights(os.path.join('models', 'generatormodel.h5'))
imgs = generator.predict(tf.random.normal((16, 128, 1)))

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1])
