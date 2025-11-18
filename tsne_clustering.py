#   Copyright 2019 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   P.S. We thank Ha and Eck [1] and Chen et al. [2] for their public source codes.
#        And the details about their work can be found below.
#
#       [1] https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn
#       [2] https://arxiv.org/abs/1709.04121
#
""" T-SNE clustering """

import os
import random
import numpy as np
import json
import matplotlib.pyplot
import pickle
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
from tqdm import tqdm
from scipy import misc
import glob
import re


def convert_white_color(img, new_color):
    data = img.getdata()

    newData = []
    for pixel in data:
        if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            # Is white pixel?
            newData.append(new_color)
        else:
            newData.append(pixel)
    img.putdata(newData)
    return img


def get_data(paths, mode='L'):
    """ read images from directory """
    images = []
    images_data = []
    for img_path in paths:
        image = Image.open(img_path).convert(mode=mode)
        data = np.array(image.getdata())
        images.append(image)
        images_data.append(data)

    return images, images_data


def get_pca_data(pca_path, png_paths):
    pca = np.loadtxt(pca_path)
    n = len(png_paths) / 3
    COLOR = [(255,252,232,200),(220,219,226,200),(243,221,226,200)]

    images = []
    for i, img_path in enumerate(png_paths):
        image = Image.open(img_path).convert(mode='RGBA')
        if i < n:
            image = convert_white_color(image, COLOR[0])
        elif i < 2*n:
            image = convert_white_color(image, COLOR[1])
        else:
            image = convert_white_color(image, COLOR[2])
        images.append(image)

    return images, pca


def applyTSNE(images_data):
    """ perform tsne clustering """
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.0, verbose=2).fit_transform(images_data)
    return tsne


def arrangeTsneByGrid(tsne, images, nx=10, ny=10, grid_width=48, grid_height=48):
    """
    arrange the tsne clustering by grid
    params:
        tsne: TSNE clustering results
        images: a list of PIL image object
        nx: the number of images put in the x-axis of the grid
        ny: the number of images put in the y-axis of the grid
        grid_width: the number of pixels (width) on a grid
        grid_height: the number of pixels (height) on a grid
    return:
        a PIL image of the clustering results
    """
    # Assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
    grid_positions, grid_size = grid_assignment
    
    full_width = grid_width * nx
    full_height = grid_height * ny
    aspect_ratio = float(grid_width) / grid_height
    
    grid_image = Image.new('RGB', (full_width, full_height), color=(255,255,255))
    
    for img, grid_pos in tqdm(zip(images, grid_positions)):
        idx_x, idx_y = grid_pos
        x, y = grid_width * idx_x, grid_height * idx_y
        img_width, img_height = img.size
        img_ar = float(img_width) / img_height 

        # Crop image for display
        if (img_ar > aspect_ratio):
            margin = 0.5 * (img_width - aspect_ratio * img_height)
            img = img.crop((int(margin), int(0), int(margin + aspect_ratio * img_height), int(img_height)))
        else:
            margin = 0.5 * (img_height - float(img_width) / aspect_ratio)
            img = img.crop((0, int(margin), int(img_width), int(margin + float(img_width) / aspect_ratio)))

        # Resize image according to grid size
        img = img.resize((int(grid_width), int(grid_height)), Image.ANTIALIAS)

        # Put the grid on the image
        grid_image.paste(img, (int(x), int(y)))

    return grid_image


def saveImage2Path(image, filepath):
    """ save a PIL Image object image to filepath """
    image.save(filepath)


def showImageOnScreen(image, width, height):
    """ show the PIL Image object image on screen with size (width, height) """
    matplotlib.pyplot.figure(figsize=(12,8))
    imshow(image)
    matplotlib.pyplot.show()


def arrangeTsneByDist(tsne, images, width=2000, height=2000, max_dim=100):
    """
    arrange the tsne clustering by distance of images in 2D space
    params:
        tsne: tsne clustering results
        image: a list of PIL Image objects
        width: the final image width (in pixel ?)
        height: the final image height (in pixel ?)
        max_dim: the max dimension size along width or height of the image (in pixel ?)
    return:
        a PIL Image object of the arrangement
    """
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    # White background
    full_image = Image.new('RGB', (width, height), color=(255,255,255))

    for img, x, y in tqdm(zip(images, tx, ty)):
        img_width, img_height = img.size
        rs = max(1, img_width/max_dim, img_height/max_dim)
        img = img.resize((int(img_width/rs), int(img_height/rs)), Image.Resampling.LANCZOS)
        full_image.paste(img, (int((width-max_dim)*x), int((height-max_dim)*y)))

    return full_image


def plotDotTsneByDist(tsne, colors, width=4000, height=2000):
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    # White background
    full_image = Image.new('RGBA', (width, height),color=(255,255,255,255))
    draw = ImageDraw.Draw(full_image)

    for x, y, c in tqdm(zip(tx, ty, colors)):
        draw.point((int(width*x),int(height*y)), fill=c)
    
    return full_image


def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths

