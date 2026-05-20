# Data-Centric Deep Learning for Noisy Labels

## Problem Statement
In real-world scenarios, perfectly annotated datasets are rarely available. Deep neural networks, with their high capacity, tend to memorize training data, including instances with incorrect labels (label noise). When trained on such corrupted datasets using standard methods, networks easily overfit to the noise, which leads to a significant degradation in generalization performance on clean test data. The objective of this project is to study this "memorization effect" and to implement and evaluate robust learning techniques capable of mitigating the impact of label noise.

## Technologies Used
- **Python 3.10**
- **PyTorch** & **Torchvision** (Deep Learning framework and models)
- **NumPy** & **Pandas** (Data manipulation)
- **Matplotlib** (Data visualization)
- **Jupyter Notebook** (Experimentation environment)

## Project Structure
The project is divided into several iterative notebooks:
- **NB1_data_noise.ipynb**: Data preparation and noise injection. Generates synthetic noisy labels to simulate real-world scenarios.
- **NB2_baseline.ipynb**: Baseline experiments. Trains a standard model to demonstrate the vulnerability to different noise levels.
- **NB3_robust_methods.ipynb**: Implementation of robust training methods.
- **NB4_analysis.ipynb**: Aggregation of results, generation of learning curves, and comparative analysis of the implemented methods.
- **nb5-corrections-ameliorations.ipynb**: Additional corrections, refinements, and improvements to the pipeline.

## Noise Injection Models
To study the effect of label noise, we artificially corrupt the CIFAR-10 dataset using two primary noise models. Let C be the total number of classes, \epsilon be the overall noise rate, y be the true label, and \tilde{y} be the corrupted label.

### 1. Symmetric Noise
Symmetric noise assumes that a true label is flipped to any other class uniformly at random. The transition probability is defined as:

```math
P(\tilde{y} = j \mid y = i) = 
\begin{cases} 
1 - \epsilon & \text{if } i = j \\ 
\frac{\epsilon}{C - 1} & \text{if } i \neq j 
\end{cases}
```

### 2. Asymmetric Noise
Asymmetric (or class-conditional) noise simulates real-world annotation errors where confusion occurs between visually similar classes (e.g., confusing a cat with a dog). Let map(i) denote the specific class that class i is commonly confused with. The transition probability is:

```math
P(\tilde{y} = j \mid y = i) = 
\begin{cases} 
1 - \epsilon & \text{if } i = j \\ 
\epsilon & \text{if } j = \text{map}(i) \\
0 & \text{otherwise}
\end{cases}
```

## Neural Network Architecture
The backbone used across all experiments is ResNet-18. Since ResNet-18 was originally designed for ImageNet (224x224 images) and CIFAR-10 contains much smaller images (32x32), the architecture has been specifically adapted to prevent aggressive spatial downsampling:

1. **First Convolutional Layer (conv1)**: 
   - Original: 7x7 kernel, stride 2.
   - Adapted: 3x3 kernel, stride 1, padding 1. This preserves the spatial resolution of the small images early in the network.
2. **Max Pooling (maxpool)**:
   - Original: 3x3 max pooling.
   - Adapted: Replaced with the identity function. Removing early pooling avoids losing critical spatial information.
3. **Fully Connected Layer (fc)**:
   - Original: 1000 output classes.
   - Adapted: Changed to a linear layer with 512 input features and 10 output classes to match CIFAR-10.

## Methods Used
We evaluate and compare four training methodologies across varying noise configurations. 

### 1. Baseline (Standard Cross-Entropy)
Trains the adapted ResNet-18 using standard Cross-Entropy Loss and SGD with Cosine Annealing.
- **Purpose**: Serves as a reference. It empirically demonstrates the memorization effect, where the training accuracy reaches near 100% while the test accuracy plummets for high noise rates.

### 2. Label Smoothing
A regularization technique that prevents the model from becoming overly confident about its predictions. 
- **Mechanism**: Instead of using hard one-hot labels, it uses soft labels. A small probability mass, denoted as alpha (typically 0.1), is subtracted from the true class and distributed uniformly among all other classes.
- **Impact**: It attenuates the gradient when a sample is incorrectly labeled, thus preventing the model from heavily penalizing itself for not perfectly matching noisy targets.

### 3. Sample Reweighting
A dynamic approach that assigns importance weights to samples during training based on their loss.
- **Mechanism**: Based on the phenomenon that neural networks learn clean, simple patterns before memorizing noise. A sample with a high loss is more likely to be noisy. We assign a weight w_i to each sample i, calculated as:

```math
w_i = \exp\left(-\frac{\mathcal{L}_i}{T}\right)
```

where L_i is the loss of the sample, and T is the Temperature parameter (set to 2.0).
- **Impact**: Noisy samples (high loss) receive a significantly lower weight, reducing their influence on the gradient updates. Weights are periodically recalculated.

### 4. Small-Loss Selection (Inspired by Co-Teaching)
An active filtering mechanism that completely drops likely noisy samples from the training batches.
- **Mechanism**: A single-network variant of the Co-Teaching method. At each epoch, the model computes the loss for all samples, sorts them in ascending order, and only trains on the top R(t) percent of samples with the lowest loss.
- **Impact**: The keep ratio R(t) progressively decays from 100% down to the estimated proportion of clean labels (1 - noise_rate). This aggressively prevents the model from updating its weights using corrupt samples.

## Experimental Configurations
All methods are systematically evaluated against 5 configurations:
1. **0% Noise (Clean)**: Absolute reference.
2. **20% Symmetric Noise**: Mild uniform noise.
3. **40% Symmetric Noise**: Moderate uniform noise (the breaking point for the baseline).
4. **60% Symmetric Noise**: Severe uniform noise (signal is dominated by noise).
5. **40% Asymmetric Noise**: Realistic, targeted noise mapping specific classes to similar ones (e.g., cat to dog, deer to horse). This simulates real-world annotation errors and is significantly harder to mitigate than symmetric noise.
