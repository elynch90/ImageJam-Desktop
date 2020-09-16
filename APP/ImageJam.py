# script for edditing the images
import numpy as np
from PIL import Image

# test = open('test_img.png')
# open the test image using PIL
test_img = Image.open('test_img.png')
# convert the test image into an array using np
test_img_array = np.asarray(test_img)

# colorize the image


def set_color(image_path, r_val, g_val, b_val):
     cur_image = Image.open(image_path)
     image_tensor = np.asarray(cur_image)
     # get the alpha values
     alpha = image_tensor[:, :, 0]
     # set red values
     r_new = image_tensor[:, :, 1] * (r_val / 200)
     # set green values
     g_new = image_tensor[:, :, 2] * (g_val / 200)
     # set blue values
     b_new = image_tensor[:, :, 3] * (b_val / 200)
     # create an image tensor based on the new color arrays
     update_array = np.array([alpha, r_new, g_new, b_new])
     image_update = Image.fromarray(update_array)
     return image_update

# change the alpha of the image


def set_alpha(image_tensor, alpha):
    # set red values
     image_tensor[:, :, 0] *= alpha
     return image_tensor

# test out the function
# print(len(sum(set_color(test_img_array, 10, 10, 10)[:, :, 2])))
