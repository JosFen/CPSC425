# problem2.py

import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from functools import partial
conv2d = partial(convolve2d, mode="same", boundary="symm")

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.

    Args:
        im1: first image
        im2: second image

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape

    # START your code here
    # HINT: You can use the conv2d defined in Line 5 for convolution operations
    # NOTE: You should remove the next three lines while coding
    # Ix = np.empty_like(im1)
    # Iy = np.empty_like(im1)
    # It = np.empty_like(im1)
    dx_kernel = np.array([[-1, 1], [-1, 1]]) * 0.5
    dy_kernel = np.array([[-1, -1], [1, 1]]) * 0.5
    dt_kernel = np.ones((2, 2)) * 0.25
    Ix = conv2d(im1, dx_kernel) + conv2d(im2, dx_kernel)
    Iy = conv2d(im1, dy_kernel) + conv2d(im2, dy_kernel)
    It = conv2d(im2, dt_kernel) - conv2d(im1, dt_kernel)
    # END your code here

    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

# consulted 
def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2.2):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: indicates whether to use Gaussian weighting
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction

    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    # START your code here
    # HINT: You can use the conv2d defined in Line 5 for convolution operations
    # NOTE: You can use either linear algebra knowledge or numpy.linalg.inv() for the matrix inverse
    # NOTE: You should remove the next two lines while coding
    # u = np.empty_like(Ix)
    # v = np.empty_like(Iy)
    half_size = patch_size // 2
    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)

    if aggregate == "gaussian":
        x = np.linspace(-half_size, half_size, patch_size)
        gauss_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        weights = np.outer(gauss_kernel, gauss_kernel)
    else:
        weights = np.ones((patch_size, patch_size))

    for i in range(half_size, Ix.shape[0] - half_size):
        for j in range(half_size, Ix.shape[1] - half_size):
            Ix_patch = Ix[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            Iy_patch = Iy[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            It_patch = It[i-half_size:i+half_size+1, j-half_size:j+half_size+1]

            A = np.stack([Ix_patch.ravel(), Iy_patch.ravel()], axis=1)
            b = -It_patch.ravel()
            # Create the diagonal matrix W from the weights
            W = np.diag(weights.ravel())
            # Compute the matrix products
            AtW = A.T @ W
            AtWA = AtW @ A
            AtWb = AtW @ b
            # Solve for the flow if the matrix is invertible
            if np.linalg.det(AtWA) > 1e-6:
                flow = np.linalg.solve(AtWA, AtWb) 
                u[i, j], v[i, j] = flow
    # END your code here
    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v


def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image
        u, v: optical flow in x and y direction

    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape

    # START your code here
    # HINT: You can use the np.meshgrid() function
    # HINT: You can use the interpolate.griddata() function with method='linear' and fill_value=0
    # NOTE: You should remove the next line while coding
    # im_warp = np.empty_like(im)
    h, w = im.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + u, 0, w - 1)
    y_new = np.clip(y + v, 0, h - 1)
    im_warp = interpolate.griddata((y.ravel(), x.ravel()), im.ravel(), (y_new.ravel(), x_new.ravel()), method='linear', fill_value=0).reshape(h, w)
    # END your code here
    assert im_warp.shape == im.shape
    return im_warp


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    # START your code here
    # NOTE: You should remove the next line while coding
    # d = 0.0
    d = np.sum((im1 - im2) ** 2) / im1.size
    # END your code here
    assert isinstance(d, float)
    return d
