import io
import os
import json
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from pycocotools.mask import encode

from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_opening, binary_closing, uniform_filter
from scipy.signal import find_peaks

from skimage import measure

import matplotlib

CMAP = matplotlib.colormaps["hsv"]
SMAP = matplotlib.colormaps["viridis"]


def histd(x, bins=255, min=0.0, max=1.0):

    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError("The dimension of input tensor should be 2 or 4.")

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / (bins - 1)
    BIN_Table = torch.arange(start=0, end=bins + 1, step=1) * delta

    for dim in range(0, bins, 1):
        h_r = BIN_Table[dim].item() if dim > 0 else 0
        h_r_sub_1 = BIN_Table[dim - 1].item()
        h_r_plus_1 = BIN_Table[dim + 1].item()

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(
            ((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1
        )
        hist_torch[:, :, dim] += torch.sum(
            ((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1
        )

    return hist_torch / hist_torch.sum(axis=-1, keepdim=True)


def discrete_phases(phases, num_bins):
    b, _, h, w = phases.shape

    angles = torch.linspace(-np.pi, np.pi, num_bins)
    phases = phases.mean(dim=1).flatten(start_dim=1)

    result = []
    for i in range(b):
        distances = []

        for angle in angles:
            distances.append(polar_distance(phases[i], angle))
        # result.append((-1 * torch.stack(distances)).argmax(dim=0).view(h, w))
        result.append((-1 * torch.stack(distances)).softmax(dim=0).view(num_bins, h, w))
    return torch.stack(result)


def plot(
    input_image,
    reconstruction,
    complex_output,
    clusters,
    cluster_labels,
    img_func,
):
    img_s, img_c, img_r, img_p, img_m, img_l = [], [], [], [], [], []

    # clusters = torch.argmax(clusters, dim=1)

    for i in range(len(input_image)):
        img = input_image[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        pha = complex_output.angle()[i].cpu().detach().mean(dim=0).numpy()
        mag = complex_output.abs()[i].cpu().detach().mean(dim=0).numpy()

        # # Plot color mapping
        # pha = np.tile(np.linspace(0, 2 * np.pi, 32), (32, 1))
        # mag = np.ones_like(pha)

        pha_norm = (pha + np.pi) / (2 * np.pi)

        pha_img = CMAP(pha_norm)

        rec_img = reconstruction[i].cpu().detach().numpy().transpose(1, 2, 0)
        rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min())

        pha_flt = pha.reshape(pha.shape[0] * pha.shape[1])
        mag_flt = mag.reshape(mag.shape[0] * mag.shape[1])
        pha_col = CMAP((pha_flt + np.pi) / (2 * np.pi))

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.scatter(pha_flt, mag_flt, c=pha_col)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches="tight")
        pol_img = np.array(Image.open(img_buf))

        plt.clf()
        plt.close()

        mas_img = SMAP(clusters[i].cpu().detach().numpy())
        lab_img = SMAP(cluster_labels[i].cpu().detach().numpy())

        img_s.append(img_func(img))
        img_c.append(img_func(pha_img))
        img_r.append(img_func(rec_img))
        img_p.append(img_func(pol_img))
        img_m.append(img_func(mas_img))
        img_l.append(img_func(lab_img))

    return img_s, img_c, img_r, img_p, img_m, img_l


def polar_distance(theta1, theta2):
    dist = abs(theta2 - theta1)
    mask = dist > np.pi
    dist[mask] = 2 * np.pi - dist[mask]

    return dist / np.pi


def soft_mas_to_hard(mas):
    mas_idx = mas.argmax(dim=1, keepdim=True)
    mas_har = torch.zeros_like(mas)
    mas_har.scatter_(1, mas_idx, 1)

    return mas_har


def indx_mas_to_hard(mas, c_dim=None):
    if c_dim == None:
        c_dim = mas.max() + 1

    mas_har = torch.zeros_like(mas).repeat(1, c_dim, 1, 1)
    mas_har.scatter_(1, mas, 1)

    return mas_har.float()


def mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:
    C1, H, W = mask1.shape
    C2, H, W = mask2.shape

    mask1 = mask1.view(C1, H * W)
    mask2 = mask2.view(C2, H * W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    return torch.where(
        union == 0,
        torch.tensor(0.0, device=mask1.device),
        intersection / union,
    )


def bipartite_mask_matching(outputs, targets):
    C1, H, W = outputs.shape
    C2, H, W = targets.shape

    assert C1 == C2, "Output channels must match target channels"

    C = -mask_iou(outputs, targets)
    _, indices = linear_sum_assignment(C)
    return torch.tensor(indices).long()


# @torch.no_grad()
# def generate_masks(model, dataset, device, root_dir, num_clusters):
#     model.eval()

#     result = {}

#     dataloader = iter(dataset)
#     for batch_idx, (x, y) in enumerate(tqdm(dataloader, total=len(dataset))):
#         x = x.to(device).float().unsqueeze(0)

#         _, _, complex_out, _ = model(x)
#         cluster_lab, cluster_chn = apply_kmeans(complex_out, n_clusters=num_clusters)

#         mask_strings = []
#         for chn in cluster_chn[0]:
#             mask_data = encode(np.asfortranarray(chn).astype(np.uint8))
#             mask_data["counts"] = mask_data["counts"].decode("ascii")
#             mask_strings.append(mask_data)

#         result[batch_idx] = mask_strings
#     return result


def apply_kmeans(complex_output, n_clusters=20, uniform_size=7, min_area_size=0.005):
    b, _, h, w = complex_output.shape
    cluster_lab = np.zeros((b, h, w))
    cluster_chn = np.zeros((b, n_clusters, h, w))

    for i in range(len(complex_output)):
        pha = complex_output.angle()[i].cpu().detach().numpy()
        pha_norm = (pha + np.pi) / (2 * np.pi)

        # phas.append(torch.tensor(CMAP(pha_norm.mean(axis=0))))

        # 1. Apply uniform filter to phases
        pha_img = np.expand_dims(pha_norm.mean(axis=0), -1)
        pha_img = uniform_filter(pha_img, size=uniform_size)

        # 1. Scale phases
        h, w, c = pha_img.shape
        pha_img = pha_img.reshape((h * w), c)
        # pha_img = MinMaxScaler().fit_transform(pha_img)

        # 2. Apply k-means to phases
        kmc_img = KMeans(n_clusters, n_init=10).fit_transform(pha_img)
        kmc_img = np.argmax(kmc_img.reshape(h, w, n_clusters), axis=-1)

        # 3. For cluster in k-means
        cls_img = []
        for kmc_idx in range(n_clusters):
            cls_img_cur = np.zeros((h, w))
            cls_img_cur[kmc_img == kmc_idx] = 1

            # 3a. Apply closing & opening to current cluster area
            cls_img_cur = binary_closing(cls_img_cur)
            cls_img_cur = binary_opening(cls_img_cur)

            # 3b. Apply labeling to current cluster area
            cls_img_cur = measure.label(cls_img_cur)

            # 3c. For cluster in cluster image
            for cls_idx in range(cls_img_cur.max()):
                # 3c1. Store are if large enough
                if (cls_img_cur == cls_idx).sum() / (w * h) > min_area_size:
                    cls_area = np.zeros((h, w))
                    cls_area[cls_img_cur == cls_idx] = 1
                    cls_img.append(cls_area)

        # 4. Apply argmax and extract channels
        cls_img = np.argmax(cls_img, axis=0)
        cls_chn = np.zeros((n_clusters, h, w))

        for cls_idx in range(n_clusters):
            cls_area = np.zeros((h, w))
            cls_area[cls_img == cls_idx] = 1
            cls_chn[cls_idx] = cls_area

        cluster_lab[i] = cls_img
        cluster_chn[i] = cls_chn

        cluster_lab[i] = kmc_img.reshape(h, w)

    return cluster_lab, cluster_chn
