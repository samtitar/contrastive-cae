import io
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import matplotlib

CMAP = matplotlib.colormaps["hsv"]


def plot(input_image, reconstruction, complex_output, img_func):
    img_s, img_c, img_r, img_p = [], [], [], []
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
        pha_img[mag < 0.1] = [1, 1, 1, 1]

        rec_img = reconstruction[i].cpu().detach().numpy().transpose(1, 2, 0)

        rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min())
        pha_flt = pha.reshape(pha.shape[0] * pha.shape[1])
        mag_flt = mag.reshape(mag.shape[0] * mag.shape[1])
        pha_col = CMAP(pha_flt / (2 * np.pi))
        keep = mag_flt > 0.1

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.scatter(pha_flt[keep], mag_flt[keep], c=pha_col[keep])

        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches="tight")
        pol_img = np.array(Image.open(img_buf))

        plt.clf()
        plt.close()

        img_s.append(img_func(img))
        img_c.append(img_func(pha_img))
        img_r.append(img_func(rec_img))
        img_p.append(img_func(pol_img))

    return img_s, img_c, img_r, img_p


def polar_distance(theta1, theta2):
    dist = theta2 - theta1
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