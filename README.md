# Monet-Style GAN • README

## Project Overview

This repository implements a Generative Adversarial Network (GAN) that learns Claude Monet’s painting style and generates new, Monet-inspired images. It uses the Kaggle “GAN Getting Started” dataset of Monet paintings and photographs to train a DCGAN (Deep Convolutional GAN) from scratch in TensorFlow/Keras.

---

## Data Description

- **monet_jpg/**  
  - 300 Monet paintings, JPEG format, 256×256 pixels, RGB  
- **photo_jpg/**  
  - 7,028 photographs, JPEG format, 256×256 pixels, RGB  

All images share the same dimensions and color channels, which simplifies the data pipeline and batching.

---

## Code Structure

1. **Imports & Configuration**  
   - Load TensorFlow, Keras layers, and standard Python libraries.  
   - Define constants: `IMG_SIZE`, `BATCH_SIZE`, `LATENT_DIM`, etc.

2. **Exploratory Data Analysis (EDA)**  
   - Count and display random samples of Monet paintings vs. photos.  
   - Plot pixel-value histograms per channel to compare color distributions.  
   - Normalize images from [0, 255] → [–1, +1] and build a performant `tf.data.Dataset` pipeline.

3. **Model Definition**  
   - **Generator**  
     - Input: 100-dim latent vector  
     - Dense → Reshape → sequence of `Conv2DTranspose` + BatchNorm + ReLU upsampling blocks  
     - Final `tanh` activation outputs 256×256 RGB images in [–1, +1]  
   - **Discriminator**  
     - Input: real or generated 256×256 images  
     - Sequence of strided `Conv2D` + LeakyReLU + Dropout downsampling blocks  
     - Flatten → Dense(sigmoid) outputs real/fake probability

4. **Training Loop**  
   - Use Adam optimizers (learning rate 1e-4, β₁=0.5) and binary cross-entropy loss  
   - In each step:  
     1. Sample a batch of real Monet images  
     2. Generate fake images from random noise  
     3. Compute generator loss (how well fakes fool the discriminator)  
     4. Compute discriminator loss (real vs. fake classification accuracy)  
     5. Apply gradients to update both networks  
   - Record per-epoch losses and save a 4×4 grid of generated samples every 10 epochs

5. **Results Analysis**  
   - Plot training loss curves for generator vs. discriminator  
   - Identify the epoch with best balance (minimal absolute difference between losses)  
   - Visually inspect sample grids to assess style quality, brush-stroke coherence, and diversity

6. **Conclusion & Next Steps**  
   - DCGAN converged to plausible Monet-style textures by epoch 10  
   - Batch normalization, LeakyReLU, and careful LR tuning were key for stability  
   - Future improvements:  
     - Experiment with Wasserstein GAN (improved gradient behavior)  
     - Apply CycleGAN for direct photo‐to‐Monet style translation  
     - Scale up to higher resolutions with progressive growing

---

