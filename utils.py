import io
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from sklearn.cluster import KMeans

# from segmentation_models_pytorch.losses import DiceLoss
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_opening, binary_closing, uniform_filter
from sklearn.preprocessing import MinMaxScaler

from skimage import measure

import matplotlib

CMAP = matplotlib.colormaps["hsv"]
SMAP = matplotlib.colormaps["viridis"]


def plot(
    input_image, reconstruction, complex_output, clusters, cluster_labels, img_func
):
    img_s, img_c, img_r, img_p, img_m, img_l = [], [], [], [], [], []

    clusters = torch.argmax(clusters, dim=1)

    for i in range(len(input_image)):
        img = input_image[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        pha = complex_output.angle()[i].cpu().detach().mean(dim=0).numpy() + np.pi
        mag = complex_output.abs()[i].cpu().detach().mean(dim=0).numpy()

        # # Plot color mapping
        # pha = np.tile(np.linspace(0, 2 * np.pi, 32), (32, 1))
        # mag = np.ones_like(pha)

        pha_norm = pha / (2 * np.pi)

        pha_img = CMAP(pha_norm)

        rec_img = reconstruction[i].cpu().detach().numpy().transpose(1, 2, 0)
        rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min())

        pha_flt = pha.reshape(pha.shape[0] * pha.shape[1])
        mag_flt = mag.reshape(mag.shape[0] * mag.shape[1])
        pha_col = CMAP(pha_flt / (2 * np.pi))

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.scatter(pha_flt, mag_flt, c=pha_col)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches="tight")
        pol_img = np.array(Image.open(img_buf))

        plt.clf()
        plt.close()

        mas_img = SMAP(MinMaxScaler().fit_transform(clusters[i].cpu().detach().numpy()))
        lab_img = SMAP(cluster_labels[i].cpu().detach().numpy())

        img_s.append(img_func(img))
        img_c.append(img_func(pha_img))
        img_r.append(img_func(rec_img))
        img_p.append(img_func(pol_img))
        img_m.append(img_func(mas_img))
        img_l.append(img_func(lab_img))

    return img_s, img_c, img_r, img_p, img_m, img_l


def polar_distance(theta1, theta2):
    dist = (theta2 - theta1).abs()
    mask = dist > np.pi
    dist[mask] = 2 * np.pi - dist[mask]

    return dist / np.pi


def polar_distance_loss(angles, boxes, similarities):
    indices = torch.combinations(torch.arange(len(boxes)))
    m_angles = torch.zeros((len(boxes),)).to(angles.device)

    for i, box in enumerate(boxes):
        m_angles[i] = angles[box[0] : box[2], box[1] : box[3]].mean()

    m_angles_pairs = m_angles[indices]
    m_polar_distance = polar_distance(m_angles_pairs[:, 0], m_angles_pairs[:, 1])

    m_similarities_pairs = 1 - similarities[indices[:, 0], indices[:, 1]].long()

    return (m_similarities_pairs - m_polar_distance).mean()


# def local_contrast_loss(angles, kernel_size=3):
#     contrast = F.unfold(angles, kernel_size)
#     indices = torch.combinations(torch.arange(contrast.shape[1]))

#     theta1 = contrast[:, indices[:, 0]]
#     theta2 = contrast[:, indices[:, 1]]
#     dist = polar_distance(theta1, theta2)  # .sum(dim=1)
#     return dist.mean()


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
    B, C1, H, W = outputs.shape
    B, C2, H, W = targets.shape

    assert C1 == C2, "Output channels must match target channels"

    indices = np.zeros((B, C1))
    for i in range(B):
        C = -mask_iou(outputs[i], targets[i])
        _, indices[i] = linear_sum_assignment(C)
    return torch.tensor(indices).long()


def apply_kmeans(complex_output, n_clusters=5, uniform_size=7, min_area_size=0.005):
    n_clusters -= 1
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
        pha_img = MinMaxScaler().fit_transform(pha_img)

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

        for cls_idx in range(cls_img.max()):
            cls_area = np.zeros((h, w))
            cls_area[cls_img == cls_idx] = 1
            cls_chn[cls_idx] = cls_area

        cluster_lab[i] = cls_img
        cluster_chn[i] = cls_chn

    return cluster_lab, cluster_chn
