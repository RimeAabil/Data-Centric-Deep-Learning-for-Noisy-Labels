# Data-Centric Deep Learning for Noisy Labels

This repository contains a comprehensive pipeline for studying and mitigating the effects of label noise in Deep Learning classification tasks. When trained on datasets with corrupted labels, standard models tend to memorize the noise, leading to significant degradation in generalization performance. This project explores robust methods to counteract this phenomenon using an adapted ResNet-18 model on the CIFAR-10 dataset.

##  Project Structure

The project is divided into several iterative notebooks:
- **`NB1_data_noise.ipynb`**: Data preparation and noise injection. Generates synthetic noisy labels (Symmetric and Asymmetric) to simulate real-world scenarios.
- **`NB2_baseline.ipynb`**: Baseline experiments. Trains a standard model to demonstrate the vulnerability (overfitting/memorization) to different noise levels.
- **`NB3_robust_methods.ipynb`**: Implementation of robust training methods.
- **`NB4_analysis.ipynb`**: Aggregation of results, generation of learning curves, and comparative analysis of the implemented methods.
- **`nb5-corrections-ameliorations.ipynb`**: Additional corrections, refinements, and improvements to the pipeline.

---

##  Neural Network Architecture

The backbone used across all experiments is **ResNet-18**. Since ResNet-18 was originally designed for ImageNet (224x224 images) and CIFAR-10 contains much smaller images (32x32), the architecture has been specifically adapted to prevent aggressive spatial downsampling:

1. **First Convolutional Layer (`conv1`)**: 
   - Original: `7x7` kernel, stride `2`.
   - **Adapted**: `3x3` kernel, stride `1`, padding `1`. This preserves the spatial resolution of the small 32x32 images early in the network.
2. **Max Pooling (`maxpool`)**:
   - Original: `3x3` max pooling.
   - **Adapted**: Replaced with `nn.Identity()`. Removing early pooling avoids losing critical spatial information.
3. **Fully Connected Layer (`fc`)**:
   - Original: 1000 output classes.
   - **Adapted**: Changed to `nn.Linear(512, 10)` to match the 10 classes of CIFAR-10.

This architecture offers a sufficient capacity to study the memorization effect, while training efficiently (~15 mins per configuration on a T4 GPU).

---

##  Methods Used

We evaluate and compare four training methodologies across varying noise configurations. 

### 1. Baseline (Standard Cross-Entropy)
Trains the adapted ResNet-18 using standard Cross-Entropy Loss and SGD with Cosine Annealing.
- **Purpose**: Serves as a reference. It empirically demonstrates the "memorization effect", where the training accuracy reaches ~100% while the test accuracy plummets for high noise rates (e.g., $\epsilon \ge 40\%$).

### 2. Label Smoothing
A regularization technique that prevents the model from becoming overly confident about its predictions. 
- **Mechanism**: Instead of using "hard" one-hot labels (1 for the target, 0 for others), it uses "soft" labels. A small probability mass $\epsilon$ (typically $0.1$) is subtracted from the true class and distributed uniformly among all other classes.
- **Impact**: It attenuates the gradient when a sample is incorrectly labeled, thus preventing the model from heavily penalizing itself for not perfectly matching noisy targets.

### 3. Sample Reweighting
A dynamic approach that assigns importance weights to samples during training based on their loss.
- **Mechanism**: Based on the phenomenon that neural networks learn clean, simple patterns before memorizing noise (Arpit et al., 2017). A sample with a high loss is more likely to be noisy. We assign a weight $w_i = \exp(-\mathcal{L}_i / T)$ (with Temperature $T=2.0$) to each sample.
- **Impact**: Noisy samples (high loss) receive a significantly lower weight, reducing their influence on the gradient updates. Weights are periodically recalculated.

### 4. Small-Loss Selection (Inspired by Co-Teaching)
An active filtering mechanism that completely drops likely noisy samples from the training batches.
- **Mechanism**: A single-network variant of the famous "Co-Teaching" method (Han et al., 2018). At each epoch, the model computes the loss for all samples, sorts them in ascending order, and only trains on the top $R(t)\%$ samples with the lowest loss.
- **Impact**: The keep ratio $R(t)$ progressively decays from 100% down to the estimated proportion of clean labels ($1 - \text{noise\_rate}$). This aggressively prevents the model from updating its weights using corrupt samples, though it might accidentally filter out difficult clean samples.

---

##  Experimental Configurations

All methods are systematically evaluated against 5 configurations:
1. **0% Noise (Clean)**: Absolute reference.
2. **20% Symmetric Noise**: Mild uniform noise.
3. **40% Symmetric Noise**: Moderate uniform noise (the breaking point for the baseline).
4. **60% Symmetric Noise**: Severe uniform noise (signal is dominated by noise).
5. **40% Asymmetric Noise**: Realistic, targeted noise mapping specific classes to similar ones (e.g., `cat` $\rightarrow$ `dog`, `deer` $\rightarrow$ `horse`). This simulates real-world annotation errors (like the Clothing1M dataset) and is significantly harder to mitigate than symmetric noise.
