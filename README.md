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

# Results for EfficientNet-B0:
Results for performing the pruning on EfficientNet-B0, including dilution rate and accuracies of each layer.
![image](https://github.com/user-attachments/assets/985fc901-f266-4c7b-87b0-2108b7dcc630)
