import spheric_mapping
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


# Generate grid
phi = np.linspace(-180, 180, 100)
theta = np.linspace(-85, 85, 100)
coords = np.meshgrid(theta, phi)  # Mesh of theta and phi angle (spherical coords)
rho = (np.sin(coords[0].flatten() * np.pi/180) - np.cos(coords[1].flatten() * np.pi/180))**2 + 2  # Distance from orig.
coords = np.transpose((rho, coords[0].flatten(), coords[1].flatten()))  # Coords in spherical coordinates
coords = spheric_mapping.spher2cart(coords)  # Coords in cartesian coordinates

# Generate the map from cartesian coordinate and distance from origin as value to be mapped.
angle_step = 1
half_window = 5
theta_angles, phi_angles, mean_vals = spheric_mapping.to_2d_map(coords, rho,
                                                                angle_step=angle_step, half_window=half_window)


fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(mean_vals, cmap="jet",
          extent=[min(phi_angles) - half_window/2, max(phi_angles) + half_window/2,
                  min(theta_angles) - half_window/2, max(theta_angles) + half_window/2]
          )
ax.set_xlabel("Phi angle (spher. coord.)")
ax.set_ylabel("Theta angle (spher. coord.)")
plt.ion()
plt.show()

mesh = pv.PolyData(coords)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars=rho, cmap="jet", point_size=10)
plotter.show_axes()
plotter.show()