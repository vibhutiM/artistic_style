This is our implementation of the A Neural Algorithm of Artistic Style https://arxiv.org/abs/1508.06576

We replicate the original papers implementation using two architecures:

- VGG_19 architecture: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.svg

- AlexNet Architecture: https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

- We also implement a simplified implementation of Combining Markov Random Fields and Convolutional Neural Networks for
Image Synthesis https://arxiv.org/pdf/1601.04589v1.pdf\



-------------------------------------
# File Structure
data
- parameters_alexnet: Pretrained Parameters for Alexnet architecture
- test: Directory for holding Testing Images

- style_transfer.ipynb: Main interface for style transfer using vgg architecture

- style_transfer_alexnet.ipynb: Main interface for style transfer using vgg architecture

- alexnet.py: Holds the alexnet class

- create_folders.sh: Creates the data folder and the data/results folder if it is not there

- empty_model.sh: Deletes the data/results contents

README.md
- vgg.py: Holds the VGG class

- vgg_load.py: Downloads the pretrained VGG network and stores that in data folde( Make sure you have data folder, otherwise will give you an error)


------------------------------------------------------

# Instructions for Running:
 
For the First Time: Run the file:
- ./create_folders.sh

This will create the required folders
Open style_transfer_vgg.ipynb or style_transfer_alexnet.ipynb

Instructions are well specified in each of the respective files:

- content_image_path=<path-to-content-image>
- style_image_path=<path-to-style-image>
- style_layers = [List of layers of network to consider for style features]
- content_layers = [Layer for content features]
Call: 
- train_style(alpha=alpha, beta=beta, n_epochs=100,filestring="PREFIX_TOFILE", MRF=False)
Set MRF=True for simplified implementation of https://arxiv.org/pdf/1601.04589v1.pdf

---------------------------------------------------------------------