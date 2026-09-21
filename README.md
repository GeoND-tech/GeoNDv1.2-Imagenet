### Overview

This repository is meant as a proof of concept for how paraboloid neurons can be used to improve the accuracy of non-transformer CNNs by replacing the output layer with a layer of paraboloid neurons.

|   Model           | Accuracy | Parameters |
| ----------------- |-------- |-----------|
| ```mobilenetv3``` - baseline   | 59.52% | 2,542,856 |
| ```mobilenetv3-pbout```         | **64.686%** | 3,567,856 |
| ```mobilenetv3-pbfeat256```         | 62.296% | **2,299,656** |


# Using paraboloid neurons to train ResNet18 models on CIFAR100 with PyTorch

Paraboloid neuron demonstration of the [GeoND Library](https://geond.tech) for [PyTorch](http://pytorch.org/) on the CIFAR100 dataset. If you are interested in trying out the library on your own datasets, please refer to the ["How to use"](https://geond.tech/geond-docs/) section of the documentation. This repository uses Version 1.1 of the GeoND Library. You can find download instructions here: [https://geond.tech/download/](https://geond.tech/download/). Adapted from [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

## Paraboloid neurons

A paraboloid neuron is a second degree neuron that only involves twice as many parameters as a linear neuron. This is achieved by using the definition of a paraboloid as the locus of points that are equidistant from a directrix hyperplane and a focal point. The decision boundary of a linear neuron is **w**<sup>T</sup>**x**=0 (**w** is the weight vector and **x** is the input point), whereas for a paraboloid neuron is (**h**<sup>T</sup>**x**)<sup>2</sup> - ||**x**-**p**||<sub>2</sub><sup>2</sup>=0 (**h** is the directrix and **p** is the focus). These are illustrated below:

![Decision boundaries](./decisionboundaries.png)

The math equations behind them with the corresponding linear neuron equations are summarized below:

![Paraboloid neuron cheat sheet](./PNCS.png)

## Requirements
- Linux only.
- Python 3.9+, use of a virtual environment recommended.
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```
- (Optional) Download the pre-trained models by running:
```
wget -i models.txt
```

## Models
- ### resnet18
Our baseline ResNet18 model. After creating the model, we make some changes to accomodate the resolution of CIFAR100 images:
```
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
```
#### Evaluation
Download the pretrained model and run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --epochs 300 --resume resnet18.pth.tar --eval true
```
#### Training from scratch
Run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1
```




- ### resnet18-paraboloidout
A ResNet18 model (modified for CIFAR100 as above) with a layer of paraboloid neurons as the output layer. After the Version 1.2 update which fixed some issues with momentum, using momentum along with ```input_factor = 1.0``` gives the best result. The training script will handle this through a command line argument.

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
model.fc = gpt.ParaboloidOutput(model.fc.in_features, model.fc.out_features, h_factor = 0.01, wd_factor = 1., grad_factor = 1., input_factor = 1.0, output_factor = 0.1, p_factor=0.0001, init = 'spotlight')
```
Note that ```ParaboloidOutput``` is the same as ```Paraboloid```, it just uses a base configuration more appropriate for output layers.

#### Evaluation
Download the pretrained model and run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --paraboloidout True --eval True --resume resnet18-paraboloidout.pth.tar
```
#### Training from scratch
Run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.7 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1 --paraboloidout true
```











## Exploration of the momentum parameter

Below is a table with the accuracies of various ```resnet18-paraboloidout``` models for different combinations of momentum values and the use of nesterov momentum. The model seems to start having issues when the momentum is too high and/or further accelerated by nesterov momentum. The best accuracy that is achieved by the pretrained model was given using a momentum of 0.7 and no nesterov.

|   Model           | Momentum | Accuracy |
| ----------------- |-------- | -------- |
| ```resnet18-paraboloidout```   |   0.1, nesterov = False   | 78.96% |
| ```resnet18-paraboloidout```   |   0.2, nesterov = False   | 78.82% |
| ```resnet18-paraboloidout```   |   0.4, nesterov = False   | 79.12% |
| ```resnet18-paraboloidout```   |   0.5, nesterov = False   | 79.14% |
| ```resnet18-paraboloidout```   |   0.5, nesterov = True   | 79.32% |
| ```resnet18-paraboloidout```   |   0.6, nesterov = False   | 79.37% |
| ```resnet18-paraboloidout```   |   0.6, nesterov = True   | 79.32% |
| ```resnet18-paraboloidout```   |   0.7, nesterov = False   | **79.56%** |
| ```resnet18-paraboloidout```   |   0.7, nesterov = True   | 79.44% |
| ```resnet18-paraboloidout```   |   0.8, nesterov = False   | 78.87% |
| ```resnet18-paraboloidout```   |   0.9, nesterov = False   | 77.46% |

## References
- Original repository: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- GeoND Library documentation: [https://geond.tech/geond-docs/](https://geond.tech/geond-docs/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)
