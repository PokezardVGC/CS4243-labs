""" CS4243 Lab 3: Feature Matching and Applications
"""
import random

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from utils import pad, unpad
import math
import cv2

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)


##### Part 1: Keypoint Detection, Description, and Matching #####

def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve,
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    # H, W = img.shape
    window = np.ones((window_size, window_size))
    # response = np.zeros((H, W))

    """ Your code starts here """

    dx = filters.sobel_h(img)
    dy = filters.sobel_v(img)
    A = convolve(np.square(dx), window, mode='constant', cval=0.0)
    B = convolve(np.multiply(dx, dy), window, mode='constant', cval=0.0)
    C = convolve(np.square(dy), window, mode='constant', cval=0.0)

    det = np.subtract(np.multiply(A, C), np.square(B))
    tr = np.add(A, C)
    response = np.subtract(det, k * np.square(tr))

    """ Your code ends here """

    return response


def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []

    """ Your code starts here """

    mean = np.mean(patch)
    sd = np.std(patch)
    feature = np.divide(np.subtract(patch, mean), (sd + 0.0001)).flatten()
    # feature.flatten()

    """ Your code ends here """

    return feature


# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0, y - (patch_size // 2)]):y + ((patch_size + 1) // 2),
                np.max([0, x - (patch_size // 2)]):x + ((patch_size + 1) // 2)]

        desc.append(desc_func(patch))

    return np.array(desc)


# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''

    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0], 3)))

    histogram = np.zeros((4, 4, 8))

    """ Your code starts here """

    dx = filters.sobel_h(patch)
    dy = filters.sobel_v(patch)

    # qn stated to precalc the weights and changing angles to range [0, 360)
    mag_weighted = np.sqrt(np.add(np.square(dx), np.square(dy))) * weights
    ori = np.arctan2(dx, dy)
    ori = np.mod((ori * 180 / math.pi) + 180, 360)

    def window(ls, x, y):
        return ls[4 * y:4 * y + 4, 4 * x:4 * x + 4]

    for x in range(4):
        for y in range(4):
            current_orientation = window(ori, x, y)
            current_weights = window(mag_weighted, x, y)

            histogram[y, x] = np.histogram(current_orientation, bins=8, range=(0, 360), weights=current_weights)[0]

    feature = histogram.flatten()
    norm = np.linalg.norm(feature)
    feature = np.divide(feature, norm)

    """ Your code ends here """

    return feature


def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []

    """ Your code starts here """

    dis = cdist(desc1, desc2, metric='euclidean')

    for i in range(len(desc1)):
        k_nearest = sorted(enumerate(dis[i, :]), key=lambda x: x[1])[:k]
        match_pairs.append((i, k_nearest))

    """ Your code ends here """

    return match_pairs


def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)

    """ Your code starts here """

    for pairs in top_2_matches:
        first = pairs[1][0]
        second = pairs[1][1]
        val = first[1] / second[1]
        if val < match_threshold:
            match_pairs.append([pairs[0], first[0]])

    """ Your code ends here """

    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs


# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)

    keypoints = np.array([(kp.pt[1], kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])

    return keypoints, descs, angles, sizes


##### Part 2: Image Stitching #####

# GIVEN
def transform_homography(src, h_matrix, getNormalized=True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1] / transformed[-1]
    transformed = transformed.transpose().astype(np.float32)

    return transformed


def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)

    """ Your code starts here """

    src = np.copy(src)
    dst = np.copy(dst)

    src = np.hstack([src, np.ones((src.shape[0], 1))])
    dst = np.hstack([dst, np.ones((dst.shape[0], 1))])

    # m=(mx, my)
    # s=(sx,sy)
    src_mx = np.mean(src[:, 0])
    src_my = np.mean(src[:, 1])
    src_sx = np.std(src[:, 0]) * (1 / np.sqrt(2))
    src_sy = np.std(src[:, 1]) * (1 / np.sqrt(2))  # should we use sd or twice sd?

    dst_mx = np.mean(dst[:, 0])
    dst_my = np.mean(dst[:, 1])
    dst_sx = np.std(dst[:, 0]) * (1 / np.sqrt(2))
    dst_sy = np.std(dst[:, 1]) * (1 / np.sqrt(2))  # should we use sd or twice sd?

    # q_i = (p_i - m) / s
    # do we need check q_i = 0? What is small engh to consider (0, 0)?
    src_q = np.divide(np.subtract(src[:, :2], np.hstack([src_mx, src_my])), src_sx)
    dst_q = np.divide(np.subtract(dst[:, :2], np.hstack([dst_mx, dst_my])), dst_sy)
    # print(sum(src_q))
    # print(sum(dst_q))

    # T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]]
    src_T = [[1 / src_sx, 0, - np.divide(src_mx, src_sx)], [0, 1 / src_sy, -np.divide(src_my, src_sy)], [0, 0, 1]]
    dst_T = [[1 / dst_sx, 0, - np.divide(dst_mx, dst_sx)], [0, 1 / dst_sy, -np.divide(dst_my, dst_sy)], [0, 0, 1]]

    # q = T p
    src_q = np.matmul(src_T, src.T).T
    dst_q = np.matmul(dst_T, dst.T).T

    # 1. For each correspondence, create 2x9 matrix Ai
    # 2. Concatenate into single 2n x 9 matrix A
    A = np.zeros((0, 9))
    for i in range(len(src_q)):
        x = src_q[i][0]
        y = src_q[i][1]
        x_prime = dst_q[i][0]
        y_prime = dst_q[i][1]
        A1 = np.array([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A2 = np.array([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        Ai = np.vstack([A1, A2])
        A = np.vstack([A, Ai])
    # print(A.shape)

    # 3. Compute SVD
    U, S, Vt = np.linalg.svd(A)

    # 4. Store singular vector of smallest singular value
    min_s = min(S)
    min_index = np.where(S == min_s)
    min_vector = Vt[min_index]

    # 5. Reshape to get H
    K = min_vector.reshape((3, 3))

    # H = inv(T') K T
    h_matrix = np.matmul(np.matmul(np.linalg.inv(dst_T), K), src_T)

    """ Your code ends here """

    return h_matrix


def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20, seed=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:, 0]]
    matched2_unpad = keypoints2[matches[:, 1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start

    """ Your code starts here """

    H = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i in range(n_iters):

        # set seed to ensure reproducable results
        np.random.seed(seed)
        rand_index = np.random.choice(N, size=n_samples, replace=False)
        keypoint_1 = matched1_unpad[rand_index]
        keypoint_2 = matched2_unpad[rand_index]

        # our H for each keypoint matches
        h = compute_homography(keypoint_1, keypoint_2)

        # Perform pespective transformation to get projected keypoints
        keypoint2_prime = transform_homography(matched1_unpad, h)

        # find index of inlier points
        diff = np.linalg.norm(keypoint2_prime - matched2_unpad, axis=1)
        index_inliers = diff < delta

        if n_inliers < len(index_inliers):
            n_inliers = len(index_inliers)
            max_inliers = index_inliers

    # return best results
    H = compute_homography(matched1_unpad[max_inliers], matched2_unpad[max_inliers])

    """ Your code ends here """

    return H, matches[max_inliers]


##### Part 3: Mirror Symmetry Detection #####

# GIVEN 
from skimage.feature import peak_local_max


def find_peak_params(hspace, params_list, window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold,
                                   min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res


# GIVEN
def angle_with_x_axis(pi, pj):
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0] - pj[0], pi[1] - pj[1]

    if x == 0:
        return np.pi / 2

    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2


# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y, x = pi[0] - pj[0], pi[1] - pj[1]
    return np.sqrt(x ** 2 + y ** 2)


def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''

    """ Your code starts here """

    res = np.zeros_like(desc)

    for i, row in enumerate(desc):

        # we get the desired 16 x 8 first
        reshape_r = np.reshape(row, (16, 8))
        reshape_ls = np.zeros_like(reshape_r)

        for j in range(12, -4, -4):
            for k in range(0, 4):
                # we copy over the desired elements
                reshape_ls[12 - j + k][0] = reshape_r[j + k][0]

                # 2nd to last element we flip
                reshape_ls[12 - j + k][1:] = np.flip(reshape_r[j + k][1:])

        res[i] = np.reshape(reshape_ls, (128,))

    # res = np.array(res)

    """ Your code ends here """

    return res


def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''

    """ Your code starts here """

    kps, descs, angles, sizes = compute_cv2_descriptor(img, method=cv2.SIFT_create())
    mir_descs = shift_sift_descriptor(descs)

    """ Your code ends here """

    return kps, descs, sizes, angles, mir_descs


def match_mirror_descriptors(descs, mirror_descs, threshold=0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []

    """ Your code starts here """

    for descriptors, top_3_descriptors in three_matches:
        # print(top_3_descriptors)
        dis_1, dis_2, dis_1_index = -1, -1, -1  # coding like in cpp....
        try:
            for curr_descriptor in top_3_descriptors:
                # for every discriptor, we check if it's the same index, if not keep since no need "flip"
                if curr_descriptor[0] != descriptors:
                    if dis_1 == -1:
                        dis_1 = curr_descriptor[1]
                        dis_1_index = curr_descriptor[0]
                    elif dis_2 == -1:
                        dis_2 = curr_descriptor[1]

            ratio = dis_1 / dis_2
            if ratio < threshold:
                match_result.append([descriptors, dis_1_index])
        except:
            continue

    match_result = np.array(match_result)

    """ Your code ends here """

    return match_result


def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []

    """ Your code starts here """

    for keypoint_1, keypoint_2 in matches:
        coord_1 = kps[keypoint_1]
        coord_2 = kps[keypoint_2]
        coord_midpt = midpoint(coord_1, coord_2)

        curr_theta = angle_with_x_axis(coord_1, coord_2)
        thetas.append(curr_theta)

        # rho = x_m * cos(theta_m) + y_m + sin(theta_m)
        curr_rho = coord_midpt[0] * math.sin(curr_theta) + coord_midpt[1] * math.cos(curr_theta)
        rhos.append(curr_rho)

    rhos = np.array(rhos)
    thetas = np.array(thetas)

    """ Your code ends here """

    return rhos, thetas


def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)

    """ Your code starts here """

    height = im_shape[0]
    width = im_shape[1]

    rho_max = np.sqrt(height ** 2 + width ** 2)

    # we add extra +1 for the "last entry"
    T = np.arange(0, 2 * math.pi + (np.pi / 180), np.pi / 180)
    R = np.arange(-rho_max, rho_max + 1, 1)
    A = np.zeros((R.shape[0], T.shape[0]))

    for i in range(len(thetas)):
        theta = thetas[i]
        rho = rhos[i]

        # increasing bin count
        index_theta = np.argmax(theta <= T)
        index_rho = np.argmax(rho <= R)

        A[index_rho][index_theta] += 1

    _, rho_val, theta_val = find_peak_params(A, [R, T], window, threshold)

    theta_val = theta_val[:num_lines]
    rho_val = rho_val[:num_lines]

    """ Your code ends here """

    return rho_val, theta_val


"""Helper functions: You should not have to touch the following functions.
"""


def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame
