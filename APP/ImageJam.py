# script for edditing the images
import numpy as np
from PIL import Image

# test = open('test_img.png')
# open the test image using PIL
# convert the test image into an array using np
# test_img_array = np.asarray(test_img)
# print(test_img_array.shape)

# colorize the image
# flag for image inversion


def set_color(image_path, r_val, g_val, b_val, alpha_val, invert_flag):
     cur_image = Image.open(str(image_path))
     image_tensor = np.asarray(cur_image)
     # set red values
     r_new = image_tensor[:, :, 0] * (1 / (r_val + 0.00001))
     # set green values
     g_new = image_tensor[:, :, 1] * (1 / (g_val + 0.00001))
     # set blue values
     b_new = image_tensor[:, :, 2] * (1 / (b_val + 0.00001))
     # get the alpha values
     #alpha_new = image_tensor[:, :, 3] * alpha_val
     # create an updated image tensor based on 100 new color arrays
     update_array = np.zeros((image_tensor.shape), dtype=np.uint8)
     update_array[:, :, 0] = r_new  # set red channel
     update_array[:, :, 1] = g_new  # set green channel
     update_array[:, :, 2] = b_new  # set blue channel
     #update_array[:, :, 3] = alpha_new  # set alpha channel
     
     # clip to RGBA limits
     update_array = np.clip(update_array, 0, 255)
     if invert_flag:
          update_array = 255 - update_array
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
