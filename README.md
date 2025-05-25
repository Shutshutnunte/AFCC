# AFCC
Code for training and pruning deep architectures using Applied Filter Cluster Connections (AFCC) as presented in the paper: Advanced deep architecture pruning using single filter performance.
Paper link: https://arxiv.org/abs/2501.12880

# Advanced Pruning
The pruning is based on pruning decision in accordance with the single filter performance extrapolated from the mechanism underlying successful deep learning.

![image](https://github.com/user-attachments/assets/1e305cf3-11e8-48b4-a9ce-3454ceb031c7)


# Data
We used the CIFAR-100 dataset in this research: https://www.cs.toronto.edu/~kriz/cifar.html

# Algorithm:
We performed the following steps for each architecture:
1. Train architecture on specified dataset
2. Extract Single Filter Performance (SFP) matrices.
3. Calculate clusters for each filter.
4. Prune the connections such as each filter connects to filters with similar cluster elements in the consecutive layer.
5. Perform a short training session.

# Results for VGG-11:
Results for performing the pruning on VGG-11, including dilution rate and accuracies of each layer.
![image](https://github.com/user-attachments/assets/01c2fec1-8af5-47ee-b303-c914aed08fe0)

# Dataset and preprocessing 
The image pixel in the CIFAR-100 dataset were normalized to the range  [-1,1] by dividing by 255 (the maximal pixel value), multiplying by 2, and subtracting 1. In all simulations, data augmentation derived from the original images was performed, by random horizontal flipping and translating up to four pixels in each direction.

# Optimization 
The cross-entropy cost function was selected for the classification task and minimized using the stochastic gradient descent algorithm. The maximal accuracy was determined by searching through the hyper-parameters (see below). Cross-validation was performed using several validation databases, each consisting of a randomly selected fifth of the training set examples. The average results were within the same standard deviation (Std) as the reported average success rates. The Nesterov momentum and L2 regularization method were applied.

# Hyper-parameters 
The hyper-parameters η (learning rate), μ (momentum constant), and α (L2 regularization) were optimized for offline learning, using a mini-batch size of 100 inputs. The learning-rate decay schedule was also optimized. A linear scheduler was applied such that it was multiplied by the decay factor, q, every Δt epochs, and is denoted below as (q,Δt). Different hyper-parameters were used for each architecture.

# VGG-16 hyper-parameters
VGG-16 was trained over 300 epochs using the following hyper-parameters to achieve maximal accuracy on CIFAR-100:
![image](https://github.com/user-attachments/assets/e35e0782-3d8a-44eb-9ce5-e94456acec29)

The decay schedule for the learning rate during training of the entire system is defined as follows:
(q,Δt)=(0.6,20)
For the training of the FC layer, η=0.01, μ=0.975, α=1e-3, with a learning rate scheduler of q=0.975 every 1 epoch, and the other weight values and biases of the architecture remained fixed.

# VGG-11 hyper-parameters
VGG-11 was trained over 300 epochs using the following hyper-parameters to achieve maximal accuracy on CIFAR-100:
![image](https://github.com/user-attachments/assets/441602f6-0c78-4a83-8335-e9c2d206a5a3)

The decay schedule for the learning rate during training of the entire system is defined as follows:
![image](https://github.com/user-attachments/assets/1efb0bb6-c7e6-4ecf-a0ea-d4bfca51351b)

For the training of the FC layer, η=0.01, μ=0.975, α=1e-3, with a learning rate scheduler of q=0.975 every 1 epoch, and the other weight values and biases of the architecture remained fixed.
