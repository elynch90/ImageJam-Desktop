# script for edditing the images
import numpy as np
from PIL import Image

# test = open('test_img.png')
# open the test image using PIL
test_img = Image.open('test_img.png')
# convert the test image into an array using np
test_img_array = np.asarray(test_img)
print(test_img_array.shape)

# colorize the image


def set_color(image_path, r_val, g_val, b_val):
     cur_image = Image.open(str(image_path))
     image_tensor = np.asarray(cur_image)
     # get the alpha values
     alpha = image_tensor[:, :, 3]
     # set red values
     r_new = image_tensor[:, :, 0] * int(r_val / 100)
     # set green values
     g_new = image_tensor[:, :, 1] * int(g_val / 100)
     # set blue values
     b_new = image_tensor[:, :, 2] * int(b_val / 100)
     # create an updated image tensor based on 100 new color arrays
     update_array = np.zeros((image_tensor.shape), dtype=np.uint8)
     update_array[:, :, 0] = r_new  # set red channel
     update_array[:, :, 1] = g_new  # set green channel
     update_array[:, :, 2] = b_new  # set blue channel
     update_array[:, :, 3] = alpha  # set alpha channel

     # print(update_array.shape)  # check the dimension of the image
     # converting the image from array using uint8 data type for pixel values
     image_update = Image.fromarray(update_array, "RGB")
     return image_update

# change the alpha of the image


def set_alpha(image_tensor, alpha):
    # set red values
     image_tensor[:, :, 0] *= alpha
     return image_tensor

# test out the function
# print(len(sum(set_color(test_img_array, 10, 10, 10)[:, :, 2])))
