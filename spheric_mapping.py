import numpy as np
from collections.abc import Iterable


def cart2spher(x: Iterable, y: Iterable, z: Iterable):
    x_ = np.array(x)
    y_ = np.array(y)
    z_ = np.array(z)

    rho = np.sqrt(x_**2 + y_**2 + z_**2)
    theta = np.arctan2(z_, np.sqrt(x_**2 + y_**2)) * 180 / np.pi  # Arguments are inversed because we want angle to start on equator
    phi = np.arctan2(y_, x_) * 180 / np.pi
    return rho, theta, phi

def sphere_mapping(theta: Iterable, phi: Iterable, val: Iterable, angle_step=10, half_window=7):
    theta_angles = np.arange(-180, 180, angle_step)
    phi_angles = np.arange(-90, 90, angle_step)
    print(len(theta_angles), len(phi_angles))
    mat_mask = []
    for theta_ in theta_angles:
        mask_theta = (np.array(theta) < (theta_ + half_window)) * (np.array(theta) > (theta_ - half_window))
        mat_mask.append([])
        for phi_ in phi_angles:
            mask_phi = (np.array(phi) < (phi_ + half_window)) * (np.array(phi) > (phi_ - half_window))
            mat_mask[-1].append(mask_theta * mask_phi)

    mat_mask = np.array(mat_mask)
    mean_vals = np.mean(mat_mask * np.array(val), axis=-1)

    return theta_angles, phi_angles, mean_vals


x = [1, 0, 0, 1]
y = [0, 1, 0, 1]
z = [0, 0, 1, 1]
val = [10, 0, 5, 1]

rho, theta, phi = cart2spher(x, y, z)
print(list(zip(theta, phi)))
theta_angles, phi_angles, mean_vals = sphere_mapping(theta, phi, val, angle_step=10, half_window=7)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(mean_vals, cmap="grey", extent=[min(phi_angles), max(phi_angles), min(theta_angles), max(theta_angles)])
plt.show()
