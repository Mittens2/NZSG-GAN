# NZSG-GAN
This code runs a GAN based on the network zero-sum framework (see [paper](https://arxiv.org/pdf/2007.05477.pdf)). The GAN consists of three networks --- a classifer, a discriminator, and a generator. The goal of NZSG-GAN is to train a generator to inteliigently integrate two different styles for image generation.

## Dependencies
* numpy
* matplotlib
* torch
* torchvision
* PIL

## Running Instruction

The script can be run with the following arguments, 

        python nzsg_app.py [-w] [-b] [-e] [-nc] [-ngpu]
        [-ncf] [-ngf]
        [-lr] [-lrC] [-be]

        -w, --workers
        -b, --batch_size
        -e, --epochs
        -nc, --num_classes
        -ngpu, --ngpu
        -ncf, --ncf     size of first convolutional layer for classifier
        -ngf, --ngf     size of first convolutional layer for generator
        -lr, --lr       learning rate for generator and discriminator
        -lrC, --lrC     learning rate for classifier
        -be, --beta     beta1 used by Adam optimizer

## Results

The code comes packaged with images from the classical painters Claude Monet and Vincent Van Gogh. When trained with the default parameters, the evolution of the output of the generator is visualized below.

<video width="320" height="240" controls>
  <source src="figs/example.mp4" type="video/mp4">
</video>