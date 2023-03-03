""" CS4243 Lab 1: Template Matching
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: Image Preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    """ Your code starts here """

    weight = (0.299, 0.587, 0.114)
    img_gray = np.empty((img.shape[0], img.shape[1]), int)

    for row in range(len(img)):
        for col in range(len(img[row])):
            red = img[row][col][0]
            green = img[row][col][1]
            blue = img[row][col][2]
            img_gray[row][col] = int(red * weight[0] + green * weight[1] + blue * weight[2])

    """ Your code ends here """
    return img_gray


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    """ Your code starts here """

    pad_image = pad_zeros(img, 1, 1, 1, 1)
    width = pad_image.shape[0]
    height = pad_image.shape[1]

    def convolution(sobel_filter):
        try:
            new_img = np.zeros((img.shape[0], img.shape[1]), int)
        except:
            print("Image needs to be in grayscale!")

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                curr_pixel = 0
                complete_frame = True
                for u in range(len(sobel_filter)):
                    for v in range(len(sobel_filter[0])):
                        x_index = i + u - 1
                        y_index = j + v - 1
                        u_convolve = 2 - u
                        v_convolve = 2 - v

                        if x_index < 0 or y_index < 0:
                            complete_frame = False
                        else:
                            curr_pixel += pad_image[x_index][y_index] * sobel_filter[u_convolve][v_convolve]
                if complete_frame:
                    # new_pixel = np.sum(np.multiply(sobel_filter, curr_frame))
                    new_img[i - 1][j - 1] = curr_pixel
        return new_img

    img_grad_h = convolution(sobelh)
    img_grad_v = convolution(sobelv)
    img_grad_d1 = convolution(sobeld1)
    img_grad_d2 = convolution(sobeld2)

    """ Your code ends here """
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    """ Your code starts here """

    for row in range(len(img)):
        for col in range(len(img[row])):
            img_pad[row + pad_height_bef][col + pad_width_bef] = img[row][col]

    """ Your code ends here """
    return img_pad.astype(img.dtype)




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """

    colours = template.shape[2] if len(template.shape) > 2 else 1
    response = np.zeros((Ho, Wo), float)  # has to be float!!
    template = template / np.sum(template)
    template_norm = np.linalg.norm(template)

    for i in range(Ho):
        for j in range(Wo):
            new_pixel = 0
            # img[][] != img[row, col]
            curr_win = img[i: i + Hk, j: j + Wk]
            window_norm = np.linalg.norm(curr_win)

            for u in range(Hk):
                for v in range(Wk):
                    for colour in range(colours):
                        curr_pixel = template[u, v, colour] * img[u + i, v + j, colour]
                        new_pixel += curr_pixel
            new_pixel *= (1 / (template_norm * window_norm))
            response[i, j] = new_pixel

    """ Your code ends here """
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """

    #colours = template.shape[2] if len(template.shape) > 2 else 1
    response = np.zeros((Ho, Wo), float)  # has to be float!!
    template = template / np.sum(template)
    template_norm = np.linalg.norm(template)

    for i in range(Ho):
        for j in range(Wo):
            new_pixel = 0
            # img[][] != img[row, col]
            curr_win = img[i: i + Hk, j: j + Wk]
            window_norm = np.linalg.norm(curr_win)

            new_pixel = np.sum(np.multiply(template, curr_win))
            new_pixel *= (1 / (template_norm * window_norm))
            response[i, j] = new_pixel

    """ Your code ends here """
    return response




def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """

    #colours = template.shape[2] if len(template.shape) > 2 else 1
    response = np.zeros((Ho * Wo, 3 * Hk * Wk))

    # window values
    for i in range(Ho):
        for j in range(Wo):
            response[Wo * i + j] = img[i:i + Hk, j:j + Wk, :].flatten()

    template_matrix = template.reshape((-1, 1))

    # we get window and template magnitude to calc norm
    window_mag = np.sqrt(np.matmul(np.square(response), np.ones(template_matrix.shape)))
    template_mag = np.linalg.norm(template_matrix)
    norm = window_mag * template_mag

    # correlation step
    response = np.matmul(response, template_matrix)

    # as usual, divide by norm and reshape
    response = response / norm
    response = response.reshape(Ho, Wo)

    """ Your code ends here """
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    
    """ Your code starts here """

    res = np.zeros(response.shape, float)
    max_val = np.max(response)
    index = np.where(response <= threshold, 0, response)
    max_coord = []
    while np.any(index):
        curr_max = np.argmax(index)
        max_pt = np.unravel_index(curr_max, res.shape)
        max_height = max_pt[0]
        max_width = max_pt[1]
        #  set surrounding area to 0
        index[max_height, max_width] = 0
        top = max_height - suppress_range[0]
        bottom = max_height + suppress_range[0]
        left = max_width - suppress_range[1]
        right = max_width + suppress_range[1]
        if top < 0:
            top = 0
        if bottom > response.shape[0]:
            bottom = response.shape[0]
        if left < 0:
            left = 0
        if right > response.shape[1]:
            right = response.shape[1]
        index[top:bottom,
        left: right] = 0
        max_coord.append(max_pt)

    #  then we change the max pts to max value
    for coord in max_coord:
        res[coord] = max_val

    """ Your code ends here """
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """

    colours = template.shape[2] if len(template.shape) > 2 else 1
    response = np.zeros((Ho, Wo), float)  # has to be float!!
    template_mean = np.mean(template)
    template = template - template_mean
    template_norm = np.linalg.norm(template)

    for i in range(Ho):
        for j in range(Wo):
            new_pixel = 0
            # img[][] != img[row, col]
            curr_win = img[i: i + Hk, j: j + Wk]
            mean_window = np.mean(curr_win)
            curr_win = curr_win - mean_window
            window_norm = np.linalg.norm(curr_win)

            new_pixel = np.sum(np.multiply(template, curr_win))
            new_pixel *= (1 / (template_norm * window_norm))
            response[i, j] = new_pixel

    """ Your code ends here """
    return response




"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_points(response, img_ori=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.circle(response, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        if img_ori is not None:
            img_ori = cv2.circle(img_ori, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)


