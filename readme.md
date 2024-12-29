# MindSpore SNN Network Implementation

This repository contains a simple implementation of a Spiking Neural Network (SNN) using MindSpore. The goal of this project is to provide a foundation for working with SNNs in MindSpore, offering basic layers and operators necessary to construct and train such models. 

## Overview

In this project, we have implemented the following:
- A spiking neural network (SNN) model using MindSpore
- Custom layers and operators required to build and train the SNN
- Several utilities and helper functions for model development

While there are some SNN frameworks and libraries out there, this implementation is specifically designed to integrate with MindSpore, which is not as widely used for SNNs yet. There may be more advanced libraries in the future, but this project aims to offer a starting point for those interested in exploring SNNs within MindSpore.

## Features

- Custom SNN layers and operators, including LIFNode, ATan.
- Example model and training scripts to demonstrate how to work with the network.
- Simple but flexible design for easy experimentation.

## Installation

To use this code, make sure you have the latest nightly version of MindSpore installed:

```bash
# Read https://www.mindspore.cn/install/ for more information
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple 
# Clone this repository
git clone https://github.com/131AIClub/ms_snn.git
cd ms_snn
```

## Usage

To train an SNN model, simply run:

```base
python -m train.py
```

This will initialize the model and start training with the provided configuration file.

For a full list of commands and configuration options, see the train.py script or refer to the documentation in the repository.
Contributing

Contributions to the repository are welcome! If you find any bugs, have suggestions for improvements, or want to add more operators or features, feel free to submit an issue or pull request.

Please ensure any new code is well-tested and follows the existing code style.
## License

This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgements

Special thanks to the MindSpore community for their ongoing work in the deep learning field.
Inspired by various research on spiking neural networks and their application to machine learning tasks.

## Disclaimer

This is a personal project, and it is currently a work in progress. There may be limitations in terms of performance or features. Feel free to open an issue if you encounter problems or have suggestions for improvements.