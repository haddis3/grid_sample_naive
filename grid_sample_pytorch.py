import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import matplotlib.pyplot as plt
import cv2
import numpy as np


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW
    # base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    # x_base = torch.arange(W).repeat(B * H).view([B, H, W])
    # y_base = torch.arange(H).repeat(B * W).view([B, W, H]).transpose(1, 2)
    base_grid = torch.stack([x_base, y_base], 1)

    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0

    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    # print(type(x))
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    # offset = torch.cat([base_grid, flow12], dim=1)
    v_grid = norm_grid(base_grid + flow12)  # BHW2

    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
        # im1_recons = x + v_grid.permute(0, 3, 1, 2).repeat(1, C/2, 1, 1)
        # im1_recons = x + v_grid.permute(0, 3, 1, 2).view(-1).repeat([C/2]).view(B, C, H, W)
        # im1_recons = x + F.interpolate(v_grid, scale_factor=[1, C/2], mode='bilinear', align_corners=True).permute(0, 3, 1, 2)

    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
        # v_grid = F.interpolate(v_grid, scale_factor=[1, C/2], mode='bilinear', align_corners=True)
        # im1_recons = x + v_grid.permute(0, 3, 1, 2)

    return im1_recons


def _ReadFlow(flow_path, w, h):

    with open(flow_path, 'rb') as f:

        data = np.fromfile(f, np.float32, count=int(w) * int(h))
        # Reshape data into 2D array (columns, rows, bands)
        return np.reshape(data, (int(h), int(w)))

def main():

    img_path = './experiment/image/Sk55npEXD_48513_363_ori.png'
    gt_path = './experiment/image/Sk55npEXD_48513_363_xiu.png'
    flowx_path = './experiment/flow/Sk55npEXD_48513_363_vx.bin'
    flowy_path = './experiment/flow/Sk55npEXD_48513_363_vy.bin'
    image = cv2.imread(img_path)
    GT = cv2.imread(gt_path)
    h, w, c = image.shape
    flow_array_x = _ReadFlow(flowx_path, w, h)
    flow_array_y = _ReadFlow(flowy_path, w, h)
    flow_array = np.array([flow_array_x, flow_array_y])

    #### grid_sample_pytorch
    img_warp = flow_warp(torch.from_numpy(np.transpose(image[None, :, :, :]/127.5-1, [0, 3, 1, 2])), torch.from_numpy(flow_array[None, :, :, :]).double())
    np_warped_image = img_warp[0].detach().cpu().numpy().transpose([1, 2, 0])
    cv2.imwrite('./experiment/output_pytorch.png', (np_warped_image + 1) * 127.5)
    # plt.figure()
    # plt.imshow(((np_warped_image + 1) * 127.5)[:, :, ::-1].astype(np.uint8))
    # plt.text(10, 70, 'grid_sample_pytorch', fontsize=15, color='green')
    # plt.show()


if __name__ == "__main__":
    main()