# problem1.py

import numpy as np
from numpy.linalg import norm

def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):

    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array

    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    # START your code here
    # NOTE: You should remove the next line while coding
    # cost_ssd = -1
    cost_ssd = np.sum((patch1 - patch2) ** 2)
    # END your code here
    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):

    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array

    Returns:
        cost_nc: the calculated NC cost as a floating point value
    """

    # START your code here
    # HINT: You can use the norm() function imported from numpy.linalg
    # NOTE: You should remove the next line while coding
    # cost_nc = -1
    patch1_flat = patch1.reshape(-1)
    patch2_flat = patch2.reshape(-1)
    # Euclidean norms of the flattened patches:
    norm_patch1 = np.linalg.norm(patch1_flat)
    norm_patch2 = np.linalg.norm(patch2_flat)
    # normalized correlation : dot product of the normalized vectors
    cost_nc = np.dot(patch1_flat, patch2_flat) / (norm_patch1 * norm_patch2)
    # END your code here
    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches:
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape

    # START your code here
    # NOTE: You should remove the next line while coding
    # cost_val = -1

    # SSD and NC costs:
    ssd_cost = cost_ssd(patch1, patch2)
    nc_cost = cost_nc(patch1, patch2)
    # weighted sum of two cost functions:
    cost_val = (1 / patch1.size) * ssd_cost + alpha * nc_cost
    # END your code here
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image

    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'

    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    # START your code here
    # HINT: You can use the np.pad() function with mode=padding_mode
    # NOTE: You should remove the next line while coding
    # padded_img = input_img.copy()

    # Compute padding size (half of the window size)
    pad_size = window_size // 2
    # Pad the image using the specified padding mode
    padded_img = np.pad(input_img, pad_size, mode=padding_mode)
    # END your code here
    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional numpy array
        padded_img_r: The padded right-view input image as 2-dimensional numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same size as the input image without padding
    """
    assert padded_img_l.ndim == 2
    assert padded_img_r.ndim == 2
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    # START your code here
    # HINT: in numpy, there is a function named argmin
    # NOTE: You should remove the next line while coding
    # disparity = padded_img_l.copy()
    height, width = padded_img_l.shape
    disparity = np.zeros((height - window_size + 1, width - window_size + 1))

    # Loop over every pixel in the image (excluding the padded border)
    for y in range(window_size // 2, height - window_size // 2):
        for x in range(window_size // 2, width - window_size // 2):
            min_cost = float('inf')
            best_disp = 0
            # search along the horizontal scan line within max_disp range
            for d in range(max_disp):
                if x - d < window_size // 2:  # stay within bounds
                    continue

                # Extract the patches from the left and right images
                patch_left = padded_img_l[y - window_size // 2:y + window_size // 2 + 1,
                                          x - window_size // 2:x + window_size // 2 + 1]
                patch_right = padded_img_r[y - window_size // 2:y + window_size // 2 + 1,
                                           x - d - window_size // 2:x - d + window_size // 2 + 1]

                # cost for this disparity and patch pair
                cost = cost_function(patch_left, patch_right, alpha)
                # minimum cost
                if cost < min_cost:
                    min_cost = cost
                    best_disp = d

            disparity[y - window_size // 2, x - window_size // 2] = best_disp
    # END your code here
    assert disparity.ndim == 2
    return disparity


def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:

    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array

    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2
    assert disparity_res.ndim == 2
    assert disparity_gt.shape == disparity_res.shape

    # START your code here
    # NOTE: You should remove the next line while coding
    # aepe = -1
    # average error (AEPE): absolute error between the ground truth and estimated disparity: 
    aepe = np.mean(np.abs(disparity_gt - disparity_res))
    # END your code here
    assert np.isscalar(aepe)
    return aepe


def optimal_alpha():
    """Return alpha that leads to the smallest EPE
    (w.r.t. other values)"""
    # TODO: You need to fix the alpha value
    # alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    # return alpha
    alpha_values = [-0.06, -0.01, 0.04, 0.1]
    best_alpha = alpha_values[0]
    min_epe = float('inf')

    # AEPE for each alpha:
    for alpha in alpha_values:
        im_l = rgb2gray(load_image("part1_left.png"))
        im_r = rgb2gray(load_image("part1_right.png"))
        disparity_gt = disparity_read("part1_gt.png")

        padded_img_l = pad_image(im_l, window_size=11)
        padded_img_r = pad_image(im_r, window_size=11)
        disparity_res = compute_disparity(padded_img_l, padded_img_r, max_disp=15, window_size=11, alpha=alpha)

        aepe = compute_aepe(disparity_gt, disparity_res)
        print(f"Current Alpha: {alpha}, Current AEPE: {aepe}")
        if aepe < min_epe:
            min_epe = aepe
            best_alpha = alpha
            
    return best_alpha
