#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Armend Ukehaxhaj"
__version__ = "0.1.2"
__license__ = "MIT"

from logzero import logger
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg


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


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, d):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = d * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img):
    weak_pixel = 75
    strong_pixel = 255
    lowth = 0.05
    highth = 0.15

    high_threshold = img.max() * highth;
    lowThreshold = high_threshold * lowth;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img):
    weak_pixel = 75
    strong_pixel = 255
    M, N = img.shape
    weak = weak_pixel
    strong = strong_pixel

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def main():
    # Main starting point of the application
    logger.info("Starting Canny Edge Detector")

    # Read the image
    img = mpimg.imread("../res/lena.png")

    # Show original and transformed image
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Canny edge detector')
    # ax1.imshow(img, cmap='gray')
    # ax2.imshow(final_img, cmap='gray')
    # plt.show(block=True)

    # load the image
    # image = Image.open(os.path.join(sys.path[0], 'lena.png'))

    # convert to greyscale
    # image = image.convert('L')

    # convert image to numpy array
    data = np.asarray(img)

    # gaussian filter value
    gaussian_mask = gaussian_kernel(5, 1.4)

    # summarize shape
    logger.info(data.shape)

    logger.info("gaussian mask is: {0}".format(gaussian_mask))

    result = convolve2d(data, gaussian_mask)

    gradientMat, thetaMat = sobel_filters(result)

    nonMaxImg = non_max_suppression(gradientMat, thetaMat)
    thresholdImg = threshold(nonMaxImg)
    img_final = hysteresis(thresholdImg)

    # create Pillow image
    image2 = Image.fromarray(img_final)

    logger.info("Convolution done, opening image")

    image2.show()


if __name__ == "__main__":
    # Start the main function
    main()
