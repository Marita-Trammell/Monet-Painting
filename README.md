# Monet-Style GAN • README
## Project Overview
- Implements a **Generative Adversarial Network (GAN)** that learns Claude Monet’s style and generates Monet-inspired images.
- Trains a **Deep Convolutional GAN (DCGAN)** from scratch in TensorFlow / Keras using Kaggle’s *“GAN Getting Started”* dataset of Monet paintings.

---
## Data Description
### `monet_jpg/`
- 300 Monet paintings — JPEG, 256 × 256, RGB
### `photo_jpg/`
- 7 028 photographs — JPEG, 256 × 256, RGB
- All images share identical dimensions and color channels, simplifying preprocessing and batching.

---
## Cleaning & Preprocessing
### 1. File Integrity
- Decode each JPEG → discard corrupt files.
- Ensure exactly three color channels (convert or drop others).

### 2. Resize
- Already 256 × 256; no resizing needed.
- *Optional for external images*
      img = tf.image.resize(img, [256, 256], method='bilinear')

### 3. Normalize
      img = tf.cast(img, tf.float32)
      img = (img / 127.5) - 1.0   # [0,255] → [-1,+1] matches tanh output

### 4. Augmentation (Optional)
- Random horizontal flips.
- ± 10° rotations.
- Brightness / contrast jitter.

---
## Pipeline
      ds = (
          tf.data.Dataset.from_tensor_slices(paths)
            .shuffle(len(paths))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
      )
      # Add .cache() if the dataset fits in RAM.

---
## Exploratory Data Analysis (EDA)
- Display 4 random Monet paintings vs. 4 photos.
- RGB pixel-value histograms comparing both domains.
- Visualize a batch of normalized images.
- **Advanced:** mean/σ “average” Monet, hue histogram, PCA clusters, brightness-contrast scatter.

---
## Model Definition
### Generator (100-D noise input)
      z → Dense(16*16*512) → Reshape(16,16,512)
      for filters in [256, 128, 64, 32]:
          x = Conv2DTranspose(filters, 5, strides=2, padding='same')(x)
          x = BatchNormalization()(x)
          x = ReLU()(x)
      output = Conv2DTranspose(3, 5, activation='tanh', padding='same')(x)

### Discriminator (256×256×3 input)
      x = Input((256,256,3))
      for filters in [32, 64, 128, 256]:
          x = Conv2D(filters, 5, strides=2, padding='same')(x)
          x = LeakyReLU(0.2)(x)
          x = Dropout(0.3)(x)
      output = Dense(1, activation='sigmoid')(Flatten()(x))

> **Rationale:** Convolutions capture textures; transposed convolutions up-sample; BatchNorm + LeakyReLU stabilize adversarial training.

---
## Training Loop
- **Optimizers:** Adam (lr = 1e-4, β₁ = 0.5) both networks
- **Loss:** Binary cross-entropy   •  **Epochs:** 10   •  **Batch:** 32
1. Sample real Monet images.
2. Generate fake images from noise.
3. Compute generator loss (fool D).
4. Compute discriminator loss (real vs. fake).
5. Apply gradients.
6. Log losses, save sample grids every 2 epochs.

---
## Hyperparameter Tuning
| Run | Latent | LR(G/D) | Batch | β₁ | Arch | Notes |
|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| A | 100 | 1e-4/1e-4 | 32 | 0.5 | DCGAN | Baseline |
| B | 100 | 5e-5/5e-5 | 32 | 0.5 | DCGAN | Smoother loss |
| C | 150 | 5e-5/5e-5 | 32 | 0.5 | DCGAN | More diversity |
| D | 100 | 5e-5/1e-4 | 32 | 0.5 | DCGAN | Balanced G/D |
| E | 100 | 5e-5/5e-5 | 32 | 0.7 | DCGAN | High momentum |
| **F** | 100 | 5e-5/5e-5 | 32 | 0.5 | **WGAN-GP** | Best stability |

---
## Results & Analysis
- **DCGAN:** oscillations, mode collapse, checkerboards.
- **WGAN-GP:** smooth losses, diverse coherent brush-strokes.

**Key points**
- Lower lr + gradient penalty critical.
- Label smoothing (real = 0.9) kept D from overpowering.
- Batch 32–64 balanced variance and speed.

---
## Conclusion & Future Work
- WGAN-GP turns a fragile GAN into a stable “painter”.
- **Next:** Progressive Growing, Spectral Norm, Self-Attention GAN, CycleGAN for photo→Monet, automated FID checkpointing.

---
## Submission
      python generate_images.py          # writes generated_images/
      zip images.zip generated_images/*.jpg
      # submit images.zip (7 000–10 000 images, 256×256)

---
## References
- Goodfellow et al., *GANs*, NeurIPS 2014
- Arjovsky et al., *Wasserstein GAN*, ICML 2017
- Karras et al., *PGGAN*, 2018
- TensorFlow DCGAN tutorial
- Kaggle *“GAN Getting Started”* competition & forums
