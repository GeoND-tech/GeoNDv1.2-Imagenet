### Overview

This repository is meant as a proof of concept for how paraboloid neurons can be used to improve the accuracy of non-transformer lightweight CNNs by replacing the output layer with a layer of paraboloid neurons or using a layer of paraboloid neurons as a feature extractor. In the latter case, the resulting network is smaller and faster than the original network while achieving higher accuracy.

|   Model           | Accuracy | Parameters |
| ----------------- |-------- |-----------|
| ```mobilenetv3``` - baseline   | 59.52% | 2,542,856 |
| ```mobilenetv3-pbout```         | **64.686%** | 3,567,856 |
| ```mobilenetv3-pbfeat256```         | 62.296% | **2,299,656** |


# Using paraboloid neurons to train MobileNetV3 models on Imagenet with PyTorch

Paraboloid neuron demonstration of the [GeoND Library](https://geond.tech) for [PyTorch](http://pytorch.org/) on the Imagenet dataset. If you are interested in trying out the library on your own datasets, please refer to the ["How to use"](https://geond.tech/geond-docs/) section of the documentation. This repository uses Version 1.2 of the GeoND Library. You can find download instructions here: [https://geond.tech/download/](https://geond.tech/download/). Adapted from [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

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
- ### mobilenetv3
Our baseline MobileNetV3 model. 

#### Evaluation
Download the pretrained model and run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.9  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp --eval True --resume mobilenetv3baseline.pth.tar
```
replacing PATHTOIMAGENET with the path that Imagenet is accessible on your system.

#### Training from scratch
Run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.9  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp
```
replacing PATHTOIMAGENET with the path that Imagenet is accessible on your system.



- ### mobilenetv3-pbout
A MobileNetV3 model with a layer of paraboloid neurons as the output layer. 

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
model.classifier = gpt.ParaboloidOutput(model.classifier.in_features, model.classifier.out_features, h_factor = 0.01, p_factor=0.0001, wd_factor = 1., grad_factor = 1., input_factor = 1., output_factor = 0.1, init = 'spotlight')
```
Note that ```ParaboloidOutput``` is the same as ```Paraboloid```, it just uses a base configuration more appropriate for output layers.

#### Evaluation
Download the pretrained model and run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.0  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp --eval True --paraboloidout True --resume mobilenetv3pbout.pth.tar
```
replacing PATHTOIMAGENET with the path that Imagenet is accessible on your system.
#### Training from scratch
Run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.0  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp --paraboloidout True
```

- ### mobilenetv3-pbfeat256

While ```mobilenetv3-pbout``` achieves higher accuracy, it significantly increases the number of model parameters. To address this issue, we made another model that reduces the dimensionality of the feature vector from 1024 to 256 using a paraboloid neuron layer. The classification layer is still linear.

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
      model.classifier = nn.Sequential(
      gpt.Paraboloid(model.classifier.in_features, 256, h_factor = 0.01, p_factor=0.0001, wd_factor = 1., input_factor = 0.1, output_factor = 0.1, grad_factor = 1., init = 'live'),
      nn.Linear(256, 1000),

```

#### Evaluation
Download the pretrained model and run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.01  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp --eval True --paraboloidout True --resume mobilenetv3pbout.pth.tar
```
replacing PATHTOIMAGENET with the path that Imagenet is accessible on your system.
#### Training from scratch
Run:
```
python train.py  --data-dir PATHTOIMAGENET   --epochs 100   --batch-size 128   --opt sgd   --lr 0.1   --momentum 0.01  --weight-decay 5e-4   --sched cosine   --warmup-epochs 5   --amp --paraboloidout True
```














## References
- Original repository: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- GeoND Library documentation: [https://geond.tech/geond-docs/](https://geond.tech/geond-docs/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)
