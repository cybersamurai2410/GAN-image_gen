# Image Generation using Deep Generative Adversarial Networks (DCGANs)
Image generation using GANs (generative adversarial networks) using Fashion MNIST dataset.

![dcgan](https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/38f48573-f68c-4d54-85b1-40da0d8618c6)

## DGAN Model Architecture
**Generator:**<br>
<img width="418" alt="image" src="https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/bb77228e-6213-495d-bdaa-bc7f3ad3d1e8">

**Discriminator:**<br>
<img width="406" alt="image" src="https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/bbda305a-0c5b-4f20-8ebd-1b06a76e85a1">

## Training
The training process of the GAN involves a two-step optimization process:

1. Training the Discriminator: During this step, the discriminator is trained using real images from the dataset and fake images generated by the generator. The discriminator learns to correctly classify real and fake images.
2. Training the Generator: In this step, the generator is trained to generate fake images that can "fool" the discriminator. The generator tries to improve its ability to generate images that are more realistic.

The two networks are trained iteratively, with the generator trying to improve its ability to generate realistic images, and the discriminator trying to become better at distinguishing between real and fake images.

## Performance Metrics
<img width="307" alt="image" src="https://github.com/cybersamurai2410/GAN-image_gen/assets/66138996/a3ee5656-2e47-42b7-aac8-b0368fe0c3e4">
