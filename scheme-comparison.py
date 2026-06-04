import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise

# test images

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

image_2019, _ = load_mat_img(img='SF2_Competition_image_2019.mat', img_info='X')
image_2020, _ = load_mat_img(img='SF2_Competition_image_2020.mat', img_info='X')
image_2021, _ = load_mat_img(img='SF2_Competition_image_2021.mat', img_info='X')
image_2023, _ = load_mat_img(img='SF2_Competition_image_2023.mat', img_info='X')
image_2024, _ = load_mat_img(img='SF2_Competition_image_2024.mat', img_info='X')
image_2025, _ = load_mat_img(img='SF2_Competition_image_2025.mat', img_info='X')

fig, axs = plt.subplots(1, 3)
plot_image(lighthouse, ax=axs[0])
plot_image(bridge, ax=axs[1])
plot_image(flamingo, ax=axs[2])

fig, axs = plt.subplots(1, 6)
plot_image(image_2019, ax = axs[0])
plot_image(image_2020, ax = axs[1])
plot_image(image_2021, ax = axs[2])
plot_image(image_2023, ax = axs[3])
plot_image(image_2024, ax = axs[4])
plot_image(image_2025, ax = axs[5])