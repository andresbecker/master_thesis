import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

@tf.function
def get_random_cell_size_ratio(mode='random_uniform', mean=0.6, stddev=0.1, lower_bound=0.4):
    """
    sample a cell size ratio at random from a given distribution. The idea is to change the size of the cell inside the image as a data augmentation technique.
    """

    if mode == 'random_uniform':
        # Select uniformly random the cell size (between 40% and 100% of the image)
        cell_img_frac = tf.random.uniform(shape=[1], minval=lower_bound, maxval=1, dtype=tf.float32)
    elif mode == 'random_normal':
        # Select uniformly random the cell size (between 40% and 100% of the image)
        cell_img_frac = tf.random.normal(shape=[1], mean=mean, stddev=stddev, dtype=tf.float32)

        # since cell_img_frac is sampled from a normal, contrain its value
        # Lower bound
        mask_tensor = (cell_img_frac > lower_bound)
        cell_img_frac = tf.where(mask_tensor, cell_img_frac, lower_bound)
        # Upper bound
        mask_tensor = (cell_img_frac < 1)
        cell_img_frac = tf.where(mask_tensor, cell_img_frac, 1)
    else:
        raise NotImplementedError()
    #print('Target cell img fraction: ', cell_img_frac)

    return cell_img_frac

@tf.function
def get_batch_masks(tensor_img):
    """
    This function extract the cell_masks of a batch stored in the last channel.
    """
    c_filter = tf.zeros(shape=[tensor_img.shape[-1]-1,1])
    c_filter = tf.concat((c_filter, tf.ones(shape=[1,1])), axis=0)

    return tensor_img @ c_filter

@tf.function
def get_min_space(x):
    """
    this function returns the min space between the cell border and the image border.
    """
    # first extract the cell mask from the last channel of the img
    cell_mask = get_batch_masks(x)

    t_x = tf.cast(tf.math.not_equal(tf.math.reduce_sum(cell_mask, axis=(1,2)), 0), dtype=tf.float32)
    t_x_low = tf.math.argmax(t_x)
    t_x_top = tf.math.argmax(tf.reverse(t_x, axis=[0]))
    t_x_min = tf.math.minimum(t_x_low, t_x_top)

    t_y = tf.cast(tf.math.not_equal(tf.math.reduce_sum(cell_mask, axis=(0,2)), 0), dtype=tf.float32)
    t_y_low = tf.math.argmax(t_y)
    t_y_top = tf.math.argmax(tf.reverse(t_y, axis=[0]))
    t_y_min = tf.math.minimum(t_y_low, t_y_top)

    return tf.cast(tf.math.minimum(t_x_min, t_y_min), dtype=tf.float32)

@tf.function
def zoom_image(tensor_img, **kwargs):

    # get min space between the cell border and the image border
    min_space = get_min_space(tensor_img)
    # get image size
    img_size = tensor_img.shape[1]
    #print(min_space)

    # Get the fraction of the origin image to crop without losing cell information
    img_size = tf.cast(img_size, dtype=tf.float32)
    frac2crop = 1 - 2 * (min_space / img_size)
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

    # get a random cell size ratio
    cell_img_frac = get_random_cell_size_ratio(**kwargs)
    #print(cell_img_frac)

    # Create temp image with cell size specified by cell_img_frac (random)
    img_size = tf.cast(img_size, dtype=tf.float32)
    temp_size = tf.cast(cell_img_frac * img_size, dtype=tf.int32)
    temp_size = tf.repeat(temp_size,2)
    tensor_img = tf.image.resize(tensor_img,
                                 size=temp_size,
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 #method=tf.image.ResizeMethod.LANCZOS5,
                                 preserve_aspect_ratio=False,
                                 antialias=False)
    #print(tensor_img.shape)

    # Resize image to original size
    img_size = tf.cast(img_size, dtype=tf.int32)
    tensor_img = tf.image.resize_with_crop_or_pad(image=tensor_img,
                                                  target_height=img_size,
                                                  target_width=img_size)

    return tensor_img

@tf.function
def apply_random_flip(image, target):
    """
    Function to apply random left or right flipping to the imput images
    image: Tensor of shape (batch_size, img_size, img_size, n_channels)
    target: number
    """
    return tf.image.random_flip_left_right(image), target

@tf.function
def apply_random_90deg_rotations(image, target):
    """
    Function to apply a random number (0, 1, 2, 3) of 90 deg rotations to the imput images
    image: Tensor of shape (batch_size, img_size, img_size, n_channels)
    target: number
    """

    # Number of 90deg rotation
    k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

    return tf.image.rot90(image, k=k), target

@tf.function
def apply_CenterZoom(image, target, **kwargs):
    """
    Function to apply random center zoom or set all cells to same size
    image: Tensor of shape (batch_size, img_size, img_size, n_channels)
    target: number
    zoom_mode: string indicating the zoom mode
    """

    if len(image.shape) == 4:
        image = tf.map_fn(fn=lambda img: zoom_image(img, **kwargs), elems=image)
    else:
        image = zoom_image(image, **kwargs)

    return image, target

@tf.function
def apply_RandomIntencity(images, targets, mean, stddev):
    """
    Function to apply random color intencity shift to each channel (only over the measured pixels). For each channel, it samples a shift factor from a normal distribution.
    image: Tensor of shape (batch_size, img_size, img_size, n_channels)
    target: number
    mean: float, distribution mean
    stddev: float, distribution mean
    """
    # Step 1. Get the mask of each cell in the batch
    images_mask = get_batch_masks(images)

    # Step 2. Sample the per-channel shift
    n_channels = images.shape[-1]
    # random shift only for the input channels
    channel_shifts = tf.random.normal(shape=[n_channels-1], mean=mean, stddev=stddev)
    # for the cell mask we add 0 to avoid modifying it
    channel_shifts = tf.concat((channel_shifts, tf.zeros(shape=[1,])), axis=0)
    channel_shifts_tensor = tf.linalg.diag(channel_shifts)

    # Step 3. Create batch containing per-channel and masked shifts
    batch_channel_shifts = tf.repeat(images_mask, repeats=n_channels, axis=-1)
    # replace the 'ones' in the masks for the random shifts
    batch_channel_shifts = batch_channel_shifts @ channel_shifts_tensor

    return images + batch_channel_shifts, targets

@tf.function
def apply_data_preprocessing(image, target, projection_tensor):
    """
    Function to preprocess cell images
    image: Tensor of shape (bath_size, img_size, img_size, n_channels)
    target: number
    projection_tensor: Tensor of shape (n_total_channels, n_selected_channels), where values for selected channels are 1 in the diagonal an 0 otherwise.
    """

    # give the image the correct data type
    image = tf.cast(image, dtype=tf.float32)

    # Remove unselected channels
    image = image @ projection_tensor

    return image, target

def get_projection_tensor(input_shape, input_ids):
    """This function returns a tensor which will be used during data preprocessing to filter channels.
    The reason why this function is not embeded in data_preprocessing function, is because tf.function does not allow iterations over tensors
    """
    n_channels = input_shape[-1]
    n_selected_channels = input_ids.shape[-1]
    projection_matrix = np.zeros(shape=(n_channels, n_selected_channels))
    for col, row in enumerate(input_ids):
        projection_matrix[row,col] = 1

    return tf.constant(projection_matrix, dtype=tf.float32)

@tf.function
def apply_data_pp_and_aug(images, targets, p, projection_tensor):

    # Preprocess data (filter channels)
    images, targets = apply_data_preprocessing(images, targets, projection_tensor)

    # Data Agmentation processes (only for train_data)
    # random Left and right flip
    if p['random_horizontal_flipping']:
        images, targets = apply_random_flip(images, targets)

    # Number of 90deg rotation
    if p['random_90deg_rotations']:
        images, targets = apply_random_90deg_rotations(images, targets)

    # ZoomIn and ZoomOut
    if p['CenterZoom']:
        images, targets = apply_CenterZoom(
                image=images,
                target=targets,
                mode=p['CenterZoom_mode'],
                mean=p['cell_size_ratio_mean'],
                stddev=p['cell_size_ratio_stddev'],
                lower_bound=p['cell_size_ratio_low_bound']
                )

    # Random channel intensity
    if p['Random_channel_intencity']:
        images, targets = apply_RandomIntencity(images, targets, p['RCI_mean'], p['RCI_stddev'])

    return images, targets

def prepare_train_and_val_TFDS_2(train_data, val_data, projection_tensor, p):

    buffer_size = 512
    BATCH_SIZE = p['BATCH_SIZE']
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data = train_data.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # the lambda function is to be able to pass more arguments to the map function
    # Train data
    train_data = train_data.map(lambda image, target: apply_data_pp_and_aug(image, target, p, projection_tensor), num_parallel_calls=AUTOTUNE)
    train_data = train_data.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    # Val data
    val_data = val_data.map(lambda image, target: apply_data_preprocessing(image, target, projection_tensor), num_parallel_calls=AUTOTUNE)
    val_data = val_data.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_data, val_data

def prepare_train_and_val_TFDS(train_data, val_data, projection_tensor, p):

    buffer_size = 512
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # the lambda function is to be able to pass more arguments to the map function

    # Shuffle train data
    train_data = train_data.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # devide by batches
    train_data = train_data.batch(p['BATCH_SIZE'])
    val_data = val_data.batch(p['BATCH_SIZE'])

    # Data Agmentation processes

    # CenterZoom most be applyed before filtering the channels. This is because this use the mask of the image, which is saved in the last channel
    # ZoomIn and ZoomOut
    if p['CenterZoom']:
        train_data = train_data.map(lambda image, target: apply_CenterZoom(image, target,
                         mode=p['CenterZoom_mode'],
                         mean=p['cell_size_ratio_mean'],
                         stddev=p['cell_size_ratio_stddev'],
                         lower_bound=p['cell_size_ratio_low_bound']
                         ), num_parallel_calls=AUTOTUNE)

	# Random channel intensity (random per-channel shift) most be applyed before filtering the channels. This is because this use the mask of the image, which is saved in the last channel
    if p['Random_channel_intencity']:
        train_data = train_data.map(lambda image, target: apply_RandomIntencity(image, target, p['RCI_mean'], p['RCI_stddev']), num_parallel_calls=AUTOTUNE)

    # random Left and right flip
    if p['random_horizontal_flipping']:
        train_data = train_data.map(lambda image, target: apply_random_flip(image, target), num_parallel_calls=AUTOTUNE)

    # Number of 90deg rotation
    if p['random_90deg_rotations']:
        train_data = train_data.map(lambda image, target: apply_random_90deg_rotations(image, target), num_parallel_calls=AUTOTUNE)

    # Remove unwanted channels
    train_data = train_data.map(lambda image, target: apply_data_preprocessing(image, target, projection_tensor), num_parallel_calls=AUTOTUNE)
    val_data = val_data.map(lambda image, target: apply_data_preprocessing(image, target, projection_tensor), num_parallel_calls=AUTOTUNE)

    return train_data.prefetch(AUTOTUNE), val_data.prefetch(AUTOTUNE)

def visualize_tensor_cell_image(tensor, title=''):
    plt.title(title)
    plt.imshow(tensor.numpy()[:,:,10:13],
               cmap=plt.cm.PiYG,
               vmin=0, vmax=1)
    plt.axis('equal')
    plt.grid(False)

def visualize_data_augmentation(tensor_image, p):
    # Visualize the original vs. random flipping and rotations
    plt_size=np.array([5,5])

    plt.figure(figsize=(plt_size[0],plt_size[1]))
    visualize_tensor_cell_image(tensor_image[0], 'Original Cell')

    if p['random_horizontal_flipping'] | p['random_90deg_rotations'] | p['CenterZoom']:
        plt.figure(figsize=(4*plt_size[0],plt_size[1]))
        for i in range(4):
            temp_tensor = copy.deepcopy(tensor_image)
            # random Left and right flip
            if p['random_horizontal_flipping']:
                temp_tensor, _ = apply_random_flip(temp_tensor, 0)

            # Number of 90deg rotation
            if p['random_90deg_rotations']:
                temp_tensor, _ = apply_random_90deg_rotations(temp_tensor, 0)

            # ZoomIn and ZoomOut
            if p['CenterZoom']:
                temp_tensor, _ = apply_CenterZoom(temp_tensor, 0,
                                    mode=p['CenterZoom_mode'],
                                    mean=p['cell_size_ratio_mean'],
                                    stddev=p['cell_size_ratio_stddev'],
                                    lower_bound=p['cell_size_ratio_low_bound']
                                    )

            # Random channel intensity
            if p['Random_channel_intencity']:
                temp_tensor, _ = apply_RandomIntencity(temp_tensor, 0, p['RCI_mean'], p['RCI_stddev'])

            plt.subplot(1,4,i+1)
            visualize_tensor_cell_image(temp_tensor[0], 'Augmented Cell')
