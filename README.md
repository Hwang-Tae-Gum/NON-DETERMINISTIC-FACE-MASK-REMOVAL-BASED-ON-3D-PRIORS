# 3D-Prior GAN-Based Face Unmasking Network

## Description

This repository implements a **non-deterministic face mask removal framework** based on 3D priors, inspired by the paper "NON-DETERMINISTIC FACE MASK REMOVAL BASED ON 3D PRIORS" by Yin et al. The approach integrates a multi-task 3D face reconstruction module with a face inpainting module to generate controllable and diverse unmasking results.

Unlike traditional deterministic methods, this framework allows **dynamic face editing** by controlling 3D shape parameters, enabling generation of faces with different expressions and mouth movements.

## Key Features

* **Non-deterministic Results:** Generate diverse face completions instead of single fixed outputs
* **3D-Guided Inpainting:** Leverages 3DMM parameters for geometrically consistent face reconstruction
* **Controllable Generation:** Modify expressions and facial features through 3D parameter manipulation
* **Robust Mask Detection:** Trained on synthetic masked faces for improved generalization

## Dataset Information

### Custom Synthetic Face Dataset
* **SimpleFaceMaskDataset:** Custom implementation generating simple synthetic faces with geometric shapes
* **Training Set:** 800 samples | **Test Set:** 200 samples  
* **Mask Types:** Surgical (rectangular), N95 (elliptical), Cloth (polygonal)
* **Face Generation:** Elliptical face shapes with basic facial features (eyes, nose, mouth)
* **Image Resolution:** 256×256 pixels
* **Normalization:** [-1, 1] range with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

## Model Architecture

### 1. N3D Module (Multi-task 3D Reconstruction)
* **Backbone:** ResNet-50 with specialized heads
* **Outputs:** 
  - Binary mask prediction (1 channel)
  - 3DMM coefficients (237 parameters)
* **Key Components:** Gated convolution for selective feature processing

### 2. Inpainting Generator
* **Architecture:** U-Net encoder-decoder with 6 residual blocks
* **Input:** 7 channels (3 masked + 1 mask + 3 rendered 3D)
* **Output:** 3-channel completed face
* **Features:** Skip connections for detail preservation

### 3. PatchGAN Discriminator
* **Architecture:** 6-layer CNN with progressive downsampling
* **Purpose:** Adversarial training for realistic face generation

### 4. ArcFace Identity Module
* **Architecture:** ResNet-50 backbone with L2-normalized embeddings
* **Purpose:** Identity preservation during face completion

## Training Strategy

### Two-Phase Training Process:

#### Phase 1: N3D Module (5,000 steps)
Train 3D reconstruction and mask segmentation jointly:
```
L₃D = L_bce + L_coef + L_photo + λ_id·L_id + λ_lm·L_lm
```

#### Phase 2: Inpainting Module (2,000 steps)
Train generator with frozen N3D module:
```
L_G = λ_pix·L_pix + λ_id·L_id + λ_tv·L_tv + λ_adv·L_adv
```

### Hyperparameters:
* **Batch Size:** 8
* **Learning Rate:** 1e-4 → 1e-5 (at midpoint)
* **Optimizer:** Adam (β₁=0.9, β₂=0.999)
* **N3D Loss Weights:** λ_id=0.1, λ_lm=0.001
* **Generator Loss Weights:** λ_pix=10, λ_id=0.1, λ_tv=0.1, λ_adv=0.01

## Key Implementation Details

### Mathematical Formulations:
* **BCE Loss:** `L_bce = -[m⊙log(m̂) + (1-m)⊙log(1-m̂)]`
* **Coefficient Loss:** `L_coef = ||ĉ - c||₁`
* **Photo Loss:** `L_photo = ||I₃D ⊙ M - I ⊙ M||₂`
* **Identity Loss:** `L_id = 1 - cosine_similarity(F(I₃D), F(I))`
* **Pixel Loss:** `L_pix = ||Î - I||₁`
* **Total Variation Loss:** `L_tv = ||∇_x Î||₂ + ||∇_y Î||₂`

### Advanced Techniques:
* **Noise Injection:** Random noise added to masked regions for robustness
* **Gated Convolution:** Learnable feature selection mechanism
* **Gradient Penalty:** Improved discriminator stability
* **Simple 3D Rendering:** Basic geometric face rendering from 3DMM coefficients

## Usage

### Quick Start:
```python
# Initialize models
n3d_model = N3D_Module(pretrained=True).to(device)
generator = InpaintingGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
arcface = SimpleArcFace().to(device)

# Two-phase training
train_n3d_phase(n3d_model, train_loader, optimizer_n3d, loss_fn, arcface)
train_inpainting_phase(n3d_model, generator, discriminator, train_loader, 
                      optimizer_g, optimizer_d, loss_fn, arcface)

# Evaluation
results = evaluate_model(n3d_model, generator, test_loader, arcface)
visualize_results(n3d_model, generator, test_loader)
```

## Evaluation Metrics

* **L1 Loss:** Pixel-level reconstruction accuracy
* **PSNR:** Peak Signal-to-Noise Ratio (dB)
* **SSIM:** Structural Similarity Index (0-1)
* **FID Score:** Fréchet Inception Distance
* **Identity Similarity:** Cosine similarity of ArcFace features

## Advantages over Existing Methods

1. **Controllability:** Unlike deterministic methods, generates diverse results through 3D parameter control
2. **Geometric Consistency:** 3D priors ensure anatomically correct face completion
3. **Robustness:** Trained on masked faces for better generalization
4. **Quality:** Superior quantitative results compared to baseline methods

## References

This implementation is based on:
* **"NON-DETERMINISTIC FACE MASK REMOVAL BASED ON 3D PRIORS"** by Xiangnan YIN, Di Huang, Liming Chen (ICIP 2022)

## Technical Contributions

* **3D-Prior Integration:** Novel use of 3DMM coefficients for guided inpainting
* **Non-deterministic Generation:** Controllable face editing through 3D parameter manipulation
* **Two-Phase Training:** Systematic separation of 3D reconstruction and image generation
* **Synthetic Dataset:** Automated generation of diverse masked face training pairs
* **Comprehensive Evaluation:** Multi-metric assessment framework for face completion quality
