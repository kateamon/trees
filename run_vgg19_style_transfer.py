# This file contains the functions needed to perform neural style transfer using VGG19 pretrained model.
# Functions are load_image, im_convert, gram_matrix, get_features, and run_vgg19_style_transfer.

from PIL import Image
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

from style_functions import load_image, im_convert, gram_matrix, get_features, init_vgg

def run_vgg19_style_transfer(content_image, style_image):
    ''' Obtain the features for style and content images,
     obtain the gram matrices for the style loss,
     initialize target image as the style image,
     set style weights for the linear combination of losses from the 5 gram matrices’ MSE,
     set the content weight and style weight (‘a’ in the style loss image above) for relative importance of the two losses, select the optimizer for back-propagation
     and set the number of steps for iterating and modifying the target image.
     '''

    # Initialize the pretrained VGG19 model in Pytorch and freeze all model parameters as we will not be training the network. Move model to cuda if NVIDIA GPUs are available.

    vgg = models.vgg19(pretrained=True).features

    # freeze VGG params to avoid chanhe
    for param in vgg.parameters():
        param.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)

     #device = torch.device('cuda')

    # load in content and style images
    content = load_image(content_image).to(device)

    # Resize style to match content, makes code easier
    style = load_image(style_image, shape=content.shape[-2:]).to(device)

    # vgg = init_vgg

    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    #initialize the target image as the content image
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1
    style_weight = 1e6

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000  # decide how many iterations to update your image (5000)

# Iteratively modify the target image while keeping the loss minimal. Modify for ‘steps’ number of steps.

    for ii in range(1, steps+1):

        # get the features from your target image
        target_features = get_features(target, vgg)

        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    final_image = im_convert(target)

    return final_image
