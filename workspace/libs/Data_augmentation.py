import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def zoom_image(tensor_img, img_size):

    def get_min_space(x):
        t_x = tf.cast(tf.math.not_equal(tf.math.reduce_sum(x, axis=(1,2)), 0),
                      dtype=tf.float32)
        t_x_low = tf.math.argmax(t_x)
        t_x_top = tf.math.argmax(tf.reverse(t_x, axis=[0]))
        t_x_min = tf.math.minimum(t_x_low, t_x_top)
        t_y = tf.cast(tf.math.not_equal(tf.math.reduce_sum(x, axis=(0,2)), 0),
                      dtype=tf.float32)
        t_y_low = tf.math.argmax(t_y)
        t_y_top = tf.math.argmax(tf.reverse(t_y, axis=[0]))
        t_y_min = tf.math.minimum(t_y_low, t_y_top)

        return tf.cast(tf.math.minimum(t_x_min, t_y_min), dtype=tf.float32)

    # get min space between the cell border and the image border
    min_space = get_min_space(tensor_img)

    # Get the fraction of the origin image to crop without losing cell information
    frac2crop = 1 - 2*(min_space/img_size[0])
    #print('Cell fraction of image: ', frac2crop)

    # Get image with the cell at its maximum size
    # Warning! function tf.image.central_crop requires frac2crop to be a fraction between 0 and 1!
    # However, this is a tf graph function, which means that all calculations here returns a tensor.
    # Therefore, you must change the origina library image_ops_impl.py for one from github which
    # supports a tensor as input:
    # wget https://raw.githubusercontent.com/tensorflow/tensorflow/b7a7f8d178254d1361d34dfc40a58b8dce48b9d7/tensorflow/python/ops/image_ops_impl.py
    # https://github.com/tensorflow/tensorflow/pull/45613/files
    tensor_img = tf.image.central_crop(tensor_img, frac2crop)
    #print('Size after cropping: ', tensor_img.shape)

    # Select uniformly random the cell size (between 40% and 100% of the image)
    cell_img_frac = tf.random.uniform(shape=[1], minval=0.4, maxval=1, dtype=tf.float32)
    #print('Target cell img fraction: ', cell_img_frac)

    # Create temp image with cell size specified by cell_img_frac (random)
    temp_size = tf.cast(cell_img_frac * img_size[0], dtype=tf.int32)
    temp_size = tf.repeat(temp_size,2)
    tensor_img = tf.image.resize(tensor_img,
                                 size=temp_size,
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 #method=tf.image.ResizeMethod.LANCZOS5,
                                 preserve_aspect_ratio=False,
                                 antialias=False)
    #print(tensor_img.shape)

    # Resize image to original size
    tensor_img = tf.image.resize_with_crop_or_pad(image=tensor_img,
                                                  target_height=img_size[0],
                                                  target_width=img_size[0])

    # Check if obtained fraction is close to target fraction
    #min_space = get_min_space(tensor_img)
    #frac2crop = tf.constant([1], dtype=tf.float32) - 2*(min_space/img_size[0])
    #print('New cell img fraction: {}\n'.format(frac2crop.numpy()))

    return tensor_img

def filter_channels(image, target, input_ids, metadata):
    """Function to discriminated undecired channels"""

    image = tf.cast(image, dtype=tf.float32)

    n_channels = metadata.features['image'].shape[-1]
    n_selected_channels = input_ids.shape[-1]

    # Create projection matrix base on selected channels
    projection_tensor = np.zeros((n_channels, n_selected_channels))
    for col, row in enumerate(input_ids):
        projection_tensor[row,col] = 1
    projection_tensor = tf.constant(projection_tensor, dtype=tf.float32)

    new_shape = image.shape[:-1]+(n_selected_channels,)

    return tf.reshape(tf.reshape(image, (-1,n_channels)) @ projection_tensor, (new_shape)), target

def augment(image, target, p, input_ids, metadata):
    """Function to augment dataset. After channel filtering, it flips
    (horizontally) and rotates (0, 90, 180, 270 degrees) randomly the images.
    """

    image, target = filter_channels(image, target, input_ids, metadata)
    img_size = metadata.features['image'].shape[:-1]

    # random Left and right flip
    if p['random_horizontal_flipping']:
        image = tf.image.random_flip_left_right(image)

    # random rotations
    # Number of 90deg rotation
    if p['random_90deg_rotations']:
        k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)

    # ZoomIn and ZoomOut
    if p['random_CenterZoom']:
        if len(image.shape) == 4:
            image = tf.map_fn(lambda image: zoom_image(image, img_size))
        else:
            image = zoom_image(image, img_size)

    return image, target

def visualize_tensor_cell_image(tensor, title=''):
    plt.title(title)
    plt.imshow(tensor.numpy()[:,:,10:13],
               cmap=plt.cm.PiYG,
               vmin=0, vmax=1)
    plt.axis('equal')
    plt.grid(False)
