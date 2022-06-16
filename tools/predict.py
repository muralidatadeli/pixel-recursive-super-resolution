import numpy as np
from skimage.io import imread, imsave
import tensorflow as tf
from collections import defaultdict

from utils import logits_2_pixel_value
from net import Net
import sys


class SuperResModel(object):

    def __init__(self, net, low_res_image, high_res_image):
        self.low_res_image = low_res_image
        self.high_res_image = high_res_image
        self.net = net


def prepare_input_and_output(low_res_input, patch_size):
    low_res_input = low_res_input[0:patch_size, 0:patch_size, ].astype("float32")
    low_res_input = np.expand_dims(low_res_input, axis=0)

    batch_size, height, width, channels = low_res_input.shape
    out_shape = [batch_size, height * 4, width * 4, channels]
    return low_res_input, out_shape


def create_network(low_res_shape, high_res_shape):
    low_res_image = tf.placeholder(dtype=tf.float32, shape=low_res_shape, name='low_res_image')
    high_res_image = tf.placeholder(dtype=tf.float32, shape=high_res_shape, name='high_res_image')
    net = Net(hr_images=high_res_image, lr_images=low_res_image, scope='prsr')

    return SuperResModel(net=net, low_res_image=low_res_image, high_res_image=high_res_image)


def enhance(model, session, low_res_input, high_res_shape, mu=1.1):
    high_res_out_image = np.zeros(shape=high_res_shape, dtype=np.float32)

    c_logits = model.net.conditioning_logits
    p_logits = model.net.prior_logits

    np_c_logits = session.run(
        c_logits,
        feed_dict={model.low_res_image: low_res_input, model.net.train: False}
    )

    batch_size, height, width, channels = high_res_shape

    # bar = progressbar.ProgressBar()
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                np_p_logits = session.run(
                    p_logits,
                    feed_dict={model.high_res_image: high_res_out_image}
                )

                sum_logits = np_c_logits[:, i, j, c * 256:(c + 1) * 256] + \
                             np_p_logits[:, i, j, c * 256:(c + 1) * 256]

                new_pixel = logits_2_pixel_value(sum_logits, mu=mu)
                high_res_out_image[:, i, j, c] = new_pixel

    return high_res_out_image


def populate_dict(pixels, x, y, x_offset, y_offset, st, end, hr_op):
    for i, hr_i in zip(range((x*4)+st, (x*4)+end), range(st, end)):
        for j, hr_j in zip(range((y*4)+st, (y*4)+end), range(st, end)):
            pixels[(i-x_offset, j-y_offset)].append([hr_op[hr_i, hr_j, 0], hr_op[hr_i, hr_j, 1], hr_op[hr_i, hr_j, 2]])
    return pixels


def get_median(pixels):
    for k, v in pixels.items():
        red = []
        green = []
        blue = []
        if len(v) == 4:
            for i in range(0, len(v)):
                red.append(v[i][0])
                green.append(v[i][1])
                blue.append(v[i][2])
            r = sorted(red)
            r = (r[1]+r[2])/2
            g = sorted(green)
            g = (g[1]+g[2])/2
            b = sorted(blue)
            b = (b[1]+b[2])/2
            pixels[k] = [(r, g, b)]
        else:
            pixels[k] = [(0, 0, 0)]
    return pixels


def get_slice_bounds(shape, middle_patch_size):
    start = int(shape[1]/2 - (middle_patch_size/2))
    end = int(shape[1]/2 + (middle_patch_size/2))
    return start, end


def save_image(image, image_path):
    image = np.uint8(image)
    imsave(image_path, image)


def predict(input_img_path, output_img_path, ckpt, patch_size):
    low_res_input = imread(input_img_path)
    lr_input, out_shape = prepare_input_and_output(low_res_input, patch_size)

    model = create_network(list(lr_input.shape), out_shape)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, ckpt)

        if low_res_input.shape == (patch_size, patch_size, 3):
            out_high_res = enhance(
                model,
                session,
                low_res_input=lr_input,
                high_res_shape=out_shape
            )
            save_image(out_high_res[0], output_img_path)

        else:
            y_offset = (53-patch_size)*4
            x_offset = 0
            pixels = defaultdict(list)
            op_img_arr = np.zeros((128, 128, 3), "float32")
            for x in range(0, 33-patch_size):
                for y in range(54-patch_size, 79-patch_size):
                    temp = low_res_input[x:x+patch_size, y:y+patch_size, ]
                    lr_input = np.expand_dims(temp, axis=0)
                    # out_high_res = enhance(
                    #     model,
                    #     session,
                    #     low_res_input=lr_input,
                    #     high_res_shape=out_shape,
                    # )
                    out_high_res = np.zeros((32, 32, 3), "float32")
                    out_high_res[:, :, ] = 255.0
                    st, end = get_slice_bounds(out_shape, 8)
                    pixels = populate_dict(pixels, x, y, x_offset, y_offset, st, end, out_high_res)

            pixels = get_median(pixels)
            for k, v in pixels.items():
                op_img_arr[k[0], k[1], :] = v[0]
            nv = np.uint8(op_img_arr)
            import matplotlib.pyplot as plt
            plt.imshow(nv)
            plt.show()
            save_image(op_img_arr, output_img_path)


if __name__ == '__main__':
    predict(sys.argv[1],sys.argv[2],sys.argv[3], 8)