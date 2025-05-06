# QUANNs: Quasi Arithmetic Neural Networks

This repository contains the official implementation of Quasi-Arithmetic Neural Networks (QUANNs) and the Neuralized Kolmogorov Mean (NKM), introduced in our research on permutation-invariant deep learning with trainable set aggregation.

QUANNs extend conventional set function models by replacing fixed pooling operations (e.g., sum, mean, max) with a highly expressive, learnable aggregation mechanism based on the Kolmogorov mean. By neuralizing this classical mathematical construct using invertible neural networks, our approach enables end-to-end learning of both representations and pooling strategies. The result is a permutation-invariant model that achieves state-of-the-art performance on several benchmark set learning tasks, while also producing embeddings more suitable for transfer learning.

## Data

Before running the code, ensure that a ./data directory exists in the project folder:
```console
mkdir -p ./data
```

## Environment Setup
To set up a Conda environment, use the provided environment.yml file:

```console
conda env create -f environment.yml
conda activate PyQUANNs
```

*Note:* This environment is configured for CPU-only execution. If you plan to run experiments on a GPU (recommended), you may need to install PyTorch and related libraries compatible with your NVIDIA driver version. Refer to the official PyTorch installation guide (`https://pytorch.org/get-started/locally/#mac-installation`) to find the appropriate versions.

## Run
The experiments can be launched by executing the respective `.py` script in the project directory:

```console
python -u experiment.py
```

## Experiment Configuration
To modify experiment parameters such as the random seed, number of replicates, and device selection (GPU or CPU), edit the corresponding `.py` script in the project directory. These scripts define the execution settings for different experiments, allowing users to customize runs according to their needs.




