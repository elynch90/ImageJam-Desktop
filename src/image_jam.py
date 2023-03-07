# script for edditing the images
from os import path, getcwd
import numpy as np
from PIL import Image
import tkinter as tk

# test = open('test_img.png')
# open the test image using PIL
# get filepath
fp = path.join(getcwd(), 'src/test_img.png')
test_img = Image.open(fp)
# convert the test image into an array using np
test_img_array = np.asarray(test_img)
print(test_img_array.shape)

# colorize the image


def set_color(image_path, r_val, g_val, b_val, alpha, invert_flag):
    cur_image = Image.open(str(image_path))
    image_tensor = np.asarray(cur_image)
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

    # invert the image
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


def image_subupdate(cur_filepath, r_val, g_val, b_val,
                    alpha, img_w, img_h, invert_flag, window):
    # pass the current slider values to the colorizer
    image_update = set_color(cur_filepath, r_val, g_val,
                             b_val, alpha, invert_flag)
    image_update.resize(size=(img_w, img_h))
    image_update.save('cur_img', format="png")
    # update the gui image
    window.Element('MAIN_IMG').update('cur_img', size=(img_w, img_h))


def save_img(cur_filepath, r_val, g_val, b_val, alpha, invert_flag, window):
    file_path = str(tk.filedialog.asksaveasfilename()) + ".png"
    cur_image = set_color(cur_filepath, r_val, g_val,
                          b_val, alpha, invert_flag)
    # save the image to the given filepath
    cur_image.save(file_path, format="png")
    window.Element("SAVEDIR").update(Text=file_path)


def upload_img(cur_filepath, r_val, g_val, b_val, alpha,
               img_w, img_h, invert_flag, window):
    image_subupdate(cur_filepath, r_val, g_val, b_val, alpha,
                    img_w, img_h, invert_flag, window)
    # display current image with resize formatiting
    # overide to default
    window.Element("rSlider").update(value="0")
    window.Element("gSlider").update(value="0")
    window.Element("bSlider").update(value="0")
    window.Element("alphaSlider").update(value="0")
    # overwrite previous color channel vals
    r_prev = 0
    g_prev = 0
    b_prev = 0


def invert_img(cur_filepath, r_val, g_val, b_val, alpha,
               img_w, img_h, invert_flag, window):
    image_subupdate(cur_filepath, r_val, g_val, b_val,
                    alpha, invert_flag, window)

def symetric_mirror(img_array):
    """Create a symetric mirror image of an image array"""
    # get the image dimensions
    img_h, img_w, _ = img_array.shape
    # get half the image relative to the width
    half_img = img_array[:, :int(img_w / 2), :]
    # transpose the image
    half_img = np.transpose(half_img, (1, 0, 2))
    # create a mirror image along the y-axis
    half_img_b = np.flip(half_img, axis=1)
    # combine the two images
    sym_mirror_img_a = np.concatenate((half_img, half_img_b), axis=1)
    # mirror the image along the x-axis
    sym_mirror_img_b = np.flip(sym_mirror_img_a, axis=0)
    # half the size of each image
    sym_mirror_img_a = sym_mirror_img_a[:int(img_w / 2), :, :]
    sym_mirror_img_b = sym_mirror_img_b[:int(img_h / 2), :, :]
    # combine the two images
    sym_mirror_img = np.concatenate((sym_mirror_img_a, sym_mirror_img_b), axis=0)
    # how can we combine the images while keeping the original aspect ratio?
    # A: we can use the original image dimensions to determine the new image dimensions
    # we can then use the new image dimensions to create a new image array
    # resize the image to the original dimensions
    # sym_mirror_img = Image.fromarray(sym_mirror_img)
    # sym_mirror_img = sym_mirror_img.resize((img_w, img_h))
    # convert the image back to an array
    # sym_mirror_img = np.asarray(sym_mirror_img)
     
    return sym_mirror_img


def kaleidoscope(img_array, n):
    """create a kaleidoscope image for n steps"""
    orginal_shape = img_array.shape
    steps = 0
    while steps < n:
        img_array = symetric_mirror(img_array)
        steps += 1
    # resize the image to the original dimensions
    # img_array = Image.fromarray(img_array)
    # img_array = img_array.resize((orginal_shape[1], orginal_shape[0]))
    # convert the image back to an array
    # img_array = np.asarray(img_array)
    return img_array


# def main():
#     kaleido_8 = kaleidoscope(img_array, 8)
#     plt.imshow(kaleido_8)

# if __name__ == "__main__":
#     main()
