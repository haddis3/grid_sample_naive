# import tensorflow as tf
import numpy as np


def grid_sample_native(img, flow):
    """Performs a backward warp of an image using the predicted flow.
        Args:
            img: single image. [height, width, channels]
            flow: Batch of flow vectors. [height, width, 2]
        Returns:
            warped: transformed image of the same shape as the input image.
    """
    height, width, channels = img.shape
    max_x = int(height - 1)
    max_y = int(width - 1)

    zero = np.zeros([], np.int32)

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = np.reshape(img, [-1, channels])
    flow_flat = np.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    # The fractional part is used to control the bilinear interpolation.
    flow_floor = np.floor(flow_flat).astype(np.int32)
    bilinear_weights = flow_flat - np.floor(flow_flat)

    # Construct base indices which are displaced with the flow
    pos_x = np.tile(np.arange(width), [height])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [1])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]

    # Compute interpolation weights for 4 adjacent pixels
    # expand to num_batch * height * width x 1 for broadcasting in add_n below
    wa = np.expand_dims((1 - xw) * (1 - yw), 1)  # top left pixel
    wb = np.expand_dims((1 - xw) * yw, 1)  # bottom left pixel
    wc = np.expand_dims(xw * (1 - yw), 1)  # top right pixel
    wd = np.expand_dims(xw * yw, 1)  # bottom right pixel

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1

    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    dim1 = width * height
    batch_offsets = np.arange(1) * dim1
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, dim1])
    base = np.reshape(base_grid, [-1])

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Ia = gather_numpy()
    Ia = np.take(im_flat, idx_a, axis=0)
    Ib = np.take(im_flat, idx_b, axis=0)
    Ic = np.take(im_flat, idx_c, axis=0)
    Id = np.take(im_flat, idx_d, axis=0)

    # tmp = wa * Ia
    warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = np.reshape(warped_flat, [height, width, channels])

    return warped

# def grid_sample_pytorch():


def _ReadFlow(flow_path, w, h):

    with open(flow_path, 'rb') as f:

        data = np.fromfile(f, np.float32, count=int(w) * int(h))
        # Reshape data into 2D array (columns, rows, bands)
        return np.reshape(data, (int(h), int(w)))


if __name__ == "__main__":

    import torch
    import cv2
    import numpy as np
    import os

    img_path = './experiment/image/Sk55npEXD_48513_363_ori.png'
    gt_path = './experiment/image/Sk55npEXD_48513_363_xiu.png'
    flowx_path = './experiment/flow/Sk55npEXD_48513_363_vx.bin'
    flowy_path = './experiment/flow/Sk55npEXD_48513_363_vy.bin'
    image = cv2.imread(img_path)
    GT = cv2.imread(gt_path)
    h, w, c = image.shape
    flow_array_x = _ReadFlow(flowx_path, w, h)
    flow_array_y = _ReadFlow(flowy_path, w, h)
    flow_array = np.transpose(np.array([flow_array_x, flow_array_y]), [1, 2, 0])

    warp_image1 = grid_sample_native(image, flow_array)
    cv2.imwrite('./experiment/output_naive.png', warp_image1.astype(np.uint8))
