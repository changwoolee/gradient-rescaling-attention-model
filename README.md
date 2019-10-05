# GRAM: Gradient Rescaling Attention Model for Data Uncertainty Estimation in Single Image Super Resolution

A official [Keras](https://keras.io/)-based implementation of GRAM.  

## Acknowledgment

This repo is heavily based on [krasserm's implementation](https://github.com/krasserm/super-resolution).  


## Table of contents

- [Getting started](#getting-started)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained models](#pre-trained-models)
- [JPEG compression](#jpeg-compression)
- [Weight normalization](#weight-normalization)
- [Other implementations](#other-implementations)
- [Limitations](#limitations)


## Dataset

If you want to [train](#training) and [evaluate](#evaluation) models, you need to download the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
and extract the downloaded archives to a directory of your choice (`DIV2K` in the following example). The resulting 
directory structure should look like:
  
    DIV2K
      DIV2K_train_HR
      DIV2K_train_LR_bicubic
        X2
        X3
        X4
      DIV2K_train_LR_unknown
        X2
        X3
        X4
      DIV2K_valid_HR
      DIV2K_valid_LR_bicubic
        ...
      DIV2K_valid_LR_unknown
        ...
          
You only need to download DIV2K archives for those downgrade operators (unknown, bicubic) and super-resolution scales
(x2, x3, x4) that you'll actually use for training. 

Before the DIV2K images can be used they must be converted to numpy arrays and stored in a separate location. Conversion 
to numpy arrays dramatically reduces image pre-processing times. Conversion can be done with the `convert.py` script: 

    python convert.py -i ./DIV2K -o ./DIV2K_BIN numpy

In this example, converted images are written to the `DIV2K_BIN` directory. By default, training and evaluation scripts 
read from this directory which can be overriden with the `--dataset` command line option. 

## Training

### GRAM

To train GRAM, we need flags `--pred_logvar`, `--attention`, and `--block_attention_gradient`.
```
python3 train.py -p=sr-resnet -s=4 --pred_logvar --attention --block_attention_gradient -o ./output/logvar_attention_grad_rescale
```

### Pixel-wise loss (MSE)

A baseline SRResNet (A SRCNN model used in [SRGAN](https://arxiv.org/abs/1609.04802) paper) model can be trained through a pixel-wise loss, or a mean-sqared error (MSE). 

```
python3 train.py -p=sr-resnet -s=4 
```

### Uncertainty loss
Unlike to the standard mean-sqared error, a neural network can learn to predict data uncertainty (or [heteroscedastic aleatoric uncertainty](https://arxiv.org/abs/1703.04977)) through an predictive variance per pixel. In this case, the neural network has two outputs: predictive mean and predictive variance.  
We observed that using this scheme directly to SISR degenerates both PSNR and SSIM.  
Anyway, the command below is for training SRResNet by the uncertainty loss.
```
python3 train.py -p=sr-resnet -s=4 --pred_logvar -o ./output/logvar
```


## Limitations

Code in this project requires the Keras Tensorflow backend.
