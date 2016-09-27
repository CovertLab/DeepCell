---
layout: page
title: Getting Started
permalink: /starting/
group: navigation
order: 4
---

## Hardware
We have been running DeepCell on a Puget systems workstation that has a 6 core Intel Xeon processor and 2 Nvidia GTX980 graphics cards. We have also run DeepCell on Stanford's Sherlock cluster that uses Nvidia GTX Titan Black graphics cards. We have not tested our code on other setups, but any computer that has a CUDA and cuDNN compatible video card should be fine.

## Docker image
A docker image is available [here](https://hub.docker.com/r/vanvalen/deepcell/). Its use requires that you have [docker](https://docs.docker.com/engine/installation/linux/ubuntulinux/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) already installed. The docker image can be installed with the command

```bash
docker pull vanvalen/deepcell
```

The ipython notebooks can then be accessed by running the command

```bash
sudo nvidia-docker run -it -p 9999:9999 vanvalen/deepcell
```
and directing your web browser to [http://localhost:9999/](http://localhost:9999/)

## Installation
While the docker container will let you get started quickly, frequent users will find it more convenient to install the software outside of a Docker container. The following installation commands assume you are working on an Ubuntu workstation (version 14.04) with CUDA 8 and cuDNN 5 (and all their dependencies) already installed. Further details about installing theano can be found at [the Theano home page](http://deeplearning.net/software/theano/install.html). We use the bleeding edge installation of Theano, as it contains a pooling function with variable strides which is necessary. Begin by running the following commands to install git and the necessary compilers.

```bash
apt-get -y update 
apt-get install -y git curl g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev
```

Next, run the following commands to instal pyenv, a package for managing Python virtual environments.

```bash
curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
CONFIGURE_OPTS=--enable-shared pyenv install 2.7.4
```

Next, clone the DeepCell Github repository by running the command

```bash
git clone https://github.com/CovertLab/DeepCell.git
```

Move to the DeepCell folder and run the following to set up the Python virtual environment

```bash
pyenv local 2.7.4
pyenv virtualenv DeepCell
pyenv local DeepCell
pip install numpy
pip install scipy
pip install scikit-learn scikit-image matplotlib palettable libtiff tifffile h5py ipython[all]
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
pip install keras pywavelets mahotas
```

Finally, we must configure keras to use the Theano backend, and instruct Theano to use the GPU by default.

```bash
mkdir ~/.keras
echo '{"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' >> ~/.keras/keras.json
echo '[global]\ndevice = gpu\nfloatX = float32' > ~/.theanorc
```

## Brief workflow overview
DeepCell was developed to segment three different types of images
* Phase microscopy images of E. coli. 
* Fluorescent images of mammalian cell nuclei
* Phase images (in combination with a fluorescent nuclear marker) of mammalian cell cytoplasms

The workflow for creating a new convolutional neural network (conv-net) to segment a microscope image is straight forward. First, an example image is turned into a training dataset. Second, a neural network model is chosen and trained to perform the segmentation. Once the training is complete, any new image can be processed with the trained conv-net to produce a segmentation prediction. These predictions are then subjected to a final refinement step to produce a segmentation mask (a binary image where 0 denotes the background and 1 denotes the object of choice)

## Making training data sets
We've annotated several training datasets that are available in the training_data folder. Because there are differences in image acquisition between laboratories (different cameras, different magnifications, etc), we recommend creating your own training dataset for cells you want to segment. To create a new training dataset, we use ImageJ and the ROI manager. One tool we found to be very useful was a Wacom Intuos Draw graphics tablet (Model # CTL490DW ~$100). We use the free hand selection tool to trace the border of each cell, adding it to the ROI manager's list of objects. When all the cells are segmented, we then created a new image (identical in size to the original) and used the draw command to create a mask of the edges. This image is then saved as a png file. The flood fill tool is then used to fill in each cell to create a mask of the cellular interior. The edge mask is subtracted from this image (using the image calculator) and the output is also saved as a png file. The convention we have been using is to name the edge mask as 'feature_0.png' and the interior mask as 'feature_1.png'

The following static [ipython notebook](/DeepCell/making_training_data.html) details how to convert the training images and their annotation into a training data set that can be used to train a conv-net. An interactive version is also available in the keras_version folder.

## Training conv-nets
The following static [ipython notebook](/DeepCell/training_convnets.html) walks you through the script that loads training data and trains a conv-net to perform image segmentation.

## Executing conv-nets
The following static [ipython notebook](/DeepCell/running_convnets.html) walks you through the script that runs a trained conv-net on a new image. It also walks you through the downstream refinement process.

## Running on Sherlock
These instructions apply to members of the Stanford community with access to Sherlock. Make sure you have access to sherlock and have followed the instructions at the [Sherlock homepage](http://sherlock.stanford.edu) to get started. In your home directory run 'nano .theanorc' and paste the following into the file:

```bash
[global]
device = gpu
floatX = float32
[blas]
ldflags = -lopenblas
```

Save the file - this step only needs to be done once. If you have been added to the Covert lab group, In the deep cell folder, run 'pyenv local DeepCell' to assign that folder the deep cell local python environment. Otherwise, you will need to make sure all of the dependencies are installed as above. To request a GPU node on sherlock, run the command 

```bash
srun -p owners --gres=gpu:gtx:1 --time=12:00:00 --pty bash -l 
```

if you have access to owners, or run

```bash
srun -p gpu --gres=gpu:gtx:1 --time=12:00:00 --pty bash -l
```

to get into the general gpu queue.

 To utilize the GPU, run the commands

```bash
module load cuda/8
module load cuDNN/v5
```

You should now be configured to use the GPUs on sherlock. If you have requested more than 1 GPU for your session use the command 

```bash
nvidia-smi -L
```

to determine the gpu name so you can pass it to theano flags (if you need to).
