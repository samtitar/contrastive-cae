import io
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from sklearn.cluster import KMeans

import matplotlib

CMAP = matplotlib.colormaps["hsv"]
SMAP = matplotlib.colormaps["viridis"]

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


def cos_similarity(theta1, theta2):
    return torch.cos(abs(theta2 - theta1))


def cos_distance(theta1, theta2):
    return -torch.cos(abs(theta2 - theta1))

def spherical_to_cartesian_coordinates(x):
    # Second dimension of x contains spherical coordinates: (r, phi_1, ... phi_n).
    num_dims = x.shape[1]
    out = torch.zeros_like(x)

    r = x[:, 0]
    phi = x[:, 1:]

    sin_component = 1
    for i in range(num_dims - 1):
        out[:, i] = r * torch.cos(phi[:, i]) * sin_component
        sin_component = sin_component * torch.sin(phi[:, i])

    out[:, -1] = r * sin_component
    return out


def phase_to_cartesian_coordinates(phase):
    # Map phases on unit-circle and transform to cartesian coordinates.
    unit_circle_phase = torch.concat(
        (torch.ones_like(phase)[:, None] + 10, phase[:, None]), dim=1
    )

    return spherical_to_cartesian_coordinates(unit_circle_phase)
    # return unit_circle_phase


def apply_kmeans(complex_output, num_clusters=5):
    b, _, h, w = complex_output.shape

    fig = plt.figure()
    fig.add_subplot(1, 2, 1, projection="polar")
    plt.scatter(
        complex_output[0].angle().mean(dim=0).flatten(),
        complex_output[0].abs().mean(dim=0).flatten(),
    )

    input_phase = (
        phase_to_cartesian_coordinates(complex_output.angle().mean(dim=1))
        .cpu()
        .detach()
        .numpy()
    )

    input_phase = rearrange(input_phase, "b p h w -> b (h w) p")
    prediction = np.zeros((b, h, w))  # + num_clusters

    # fig.add_subplot(1, 2, 2, projection="polar")
    fig.add_subplot(1, 2, 2)
    # plt.scatter(input_phase[0, 1].flatten(), input_phase[0, 0].flatten())
    plt.scatter(input_phase[0, :, 0], input_phase[0, :, 1])
    plt.show()

    # Run k-means on each image separately.
    for img_idx in range(b):
        k_means = KMeans(n_clusters=num_clusters).fit(input_phase[img_idx])
        prediction[img_idx] = k_means.labels_.reshape(h, w)

    return prediction
