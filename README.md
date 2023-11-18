# Image Generation using Deep Generative Adversarial Networks (DCGANs)
Image generation using GANs (generative adversarial networks) using Fashion MNIST dataset.

![dcgan](https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/38f48573-f68c-4d54-85b1-40da0d8618c6)

## DGAN Model Architecture
**Generator:**
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 12544)             1618176   
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 12544)             0         
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 256)         0         
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 14, 14, 256)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 14, 14, 256)       590080    
_________________________________________________________________
batch_normalization (BatchNo (None, 14, 14, 256)       1024      
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 14, 14, 256)       0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 128)       295040    
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 128)       512       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 64)        73792     
_________________________________________________________________
batch_normalization_2 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 28, 28, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 1)         577       
=================================================================
Total params: 2,579,457
Trainable params: 2,578,561
Non-trainable params: 896
_________________________________________________________________

**Discrimintor:**
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 14, 14, 32)        320       
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 14, 14, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 7, 64)          18496     
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 64)          256       
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 7, 7, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 4, 4, 128)         73856     
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 128)         512       
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 4, 4, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
batch_normalization_5 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
leaky_re_lu_7 (LeakyReLU)    (None, 4, 4, 256)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4097      
=================================================================
Total params: 393,729
Trainable params: 392,833
Non-trainable params: 896
_________________________________________________________________


## Training
The training process of the GAN involves a two-step optimization process:

1. Training the Discriminator: During this step, the discriminator is trained using real images from the dataset and fake images generated by the generator. The discriminator learns to correctly classify real and fake images.
2. Training the Generator: In this step, the generator is trained to generate fake images that can "fool" the discriminator. The generator tries to improve its ability to generate images that are more realistic.

The two networks are trained iteratively, with the generator trying to improve its ability to generate realistic images, and the discriminator trying to become better at distinguishing between real and fake images.

## Performance Metrics
<img width="307" alt="image" src="https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/a3ee5656-2e47-42b7-aac8-b0368fe0c3e4">
