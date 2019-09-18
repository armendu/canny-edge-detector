#!/usr/bin/env python3
"""
Module Docstring
"""
import os
import sys

__author__ = "Armend Ukehaxhaj"
__version__ = "0.1.1"
__license__ = "MIT"

from logzero import logger
from PIL import Image
import numpy as np


def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    # Get kernel shape
    m, n = kernel.shape

    # Set info of image
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # convolution output
    # Add zero padding to the input image
    # image_padded = np.array(np.zeros((image.shape[0] + 2, image.shape[1] + 2)))
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    for i in range(y):  # Loop over every pixel of the image
        for j in range(x):
            # element-wise multiplication of the kernel and the image
            output[i, j] = np.sum(kernel * image_padded[i:i + m, j:j + m])
    return output


def main():
    """ Main entry point of the app """
    logger.info("Starting Canny Edge Detector")

    # load the image
    image = Image.open(os.path.join(sys.path[0], 'lena-colored.png'))

    # convert to greyscale
    image = image.convert('L')

    # convert image to numpy array
    data = np.asarray(image)

    # gaussian filter value
    gaussian_mask = gaussian_kernel(5, 1.4)

    # summarize shape
    logger.info(data.shape)

    logger.info("gaussian mask is: {0}".format(gaussian_mask))

    result = convolve2d(data, gaussian_mask)

    # create Pillow image
    image2 = Image.fromarray(result)

    logger.info("Convolution done")

    image2.show()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
