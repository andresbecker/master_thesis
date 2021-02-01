import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_cell(img, cmap='PiYG', title='', vmin=0, vmax=1, colorbar=False, alpha=1):
    fig = plt.imshow(img,
               cmap=getattr(plt.cm, cmap),
               vmin=vmin, vmax=vmax,
               aspect='equal',
               alpha=alpha
               )
    plt.title(title, fontsize=20)

    if colorbar:
        plt.colorbar(fig, orientation="vertical", pad=0.05)
        #plt.colorbar(fig)

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta

    return images

def compute_gradients(images, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)

    return tape.gradient(prediction, images)

def get_integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients

@tf.function
def get_integrated_gradients(baseline, image, model, m_steps=50, batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                           model=model)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = get_integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

#@tf.function
def get_VarGrad(img=None, img_mask=None, baseline='black', model=None, n_images=15, IG_m_steps=30):

    if baseline == 'black':
        bl = tf.zeros(shape=img.shape)
    elif baseline == 'noise':
        bl = tf.random.normal(shape=img.shape, mean=0, stddev=1)
    else:
        raise NotImplementedError('{} baseline method not implemented'.format(baseline))

    for i in range(n_images):
        rnorm_img = tf.random.normal(shape=img.shape, mean=0, stddev=1)

        # If image mask is given, then only add noice into measured pixels
        if img_mask is not None:
            rnorm_img = tf.where(img_mask, rnorm_img, [0])

        ig_temp = get_integrated_gradients(baseline=bl,
                                       image=img + rnorm_img,
                                       model=model,
                                       m_steps=IG_m_steps)

        ig_temp = tf.expand_dims(ig_temp, axis=0)

        if i == 0:
            ig_sample = ig_temp
        else:
            ig_sample = tf.concat((ig_sample, ig_temp), axis=0)

    return tf.math.reduce_std(ig_sample, axis=0)

def plot_VarGrad_IG(img=None, img_mask=None, score_map=None, top_percent=0.2, img_size=(7,7), channels_df=None, score_map_same_sacale=True):

    temp_score = score_map.numpy()
    n_channels = img.shape[-1]

    # get the total variance corresponding to each channel
    score_channel_std = []
    for c in range(n_channels):
        score_channel_std.append(np.sum(temp_score[:,:,c]))
    total_var = np.sum(score_channel_std)

    # get the the top_percent pixels (make the rest 0)
    original_shape = temp_score.shape
    temp_score = temp_score.reshape(-1)
    n_pixels = temp_score.shape[0]

    # If img mask given, then return the top MEASURED pixels
    if img_mask is None:
        cell_prop = 1
    else:
        cell_prop = np.sum(img_mask)*n_channels / np.prod(img.shape, axis=0)
    top_percent = cell_prop * top_percent
    n_top_pixels = int(top_percent * n_pixels)

    # get indices corresponding to the top pixels
    top_idxs = temp_score.argsort()[-n_top_pixels:]
    mask = np.zeros(n_pixels)
    mask[top_idxs] = 1
    mask = mask.astype(np.bool)
    # make all non top pixels equal to zero
    temp_score[~mask] = 0
    # get min and max values for plot
    if score_map_same_sacale:
        vmin = temp_score[mask].min()
        vmax = temp_score[mask].max()
    else:
        vmin = None
        vmax = None

    temp_score = temp_score.reshape(original_shape)

    plt.figure(figsize=(n_channels*img_size[0], 2*img_size[1]))

    # plot sorting by channel var percent
    #for i, c in enumerate(np.argsort(-np.array(score_channel_std)), 1):
    for c in range(n_channels):
        i = c+1
        channel_name = channels_df['name'][channels_df.channel_id == c].values[0]

        # Plot original image
        plt.subplot(3, n_channels, i)
        plot_cell(img=img[:,:,c], cmap='Blues', colorbar=True, title=channel_name)

        # Plot VarGrad_IG map (importance map)
        channel_std_percen = round(100*score_channel_std[c] / total_var, 3)
        title='Channel var: ' + str(channel_std_percen) + '%'
        plt.subplot(3, n_channels, n_channels+i)
        plot_cell(img=temp_score[:,:,c], cmap='Oranges', colorbar=True, title=title,
            vmin=vmin, vmax=vmax
            )

        # Plot image and importance map together
        plt.subplot(3, n_channels, 2*n_channels+i)
        plot_cell(img=temp_score[:,:,c], cmap='Oranges',
            vmin=vmin, vmax=vmax,
            colorbar=True
            )
        plot_cell(img=img[:,:,c], cmap='Blues', title='Cell and Score map overlapping', alpha=0.4)

    plt.show()

def plot_VarGrad_IG_with_control(img=None, img_mask=None, score_map_1=None, score_map_2=None, top_percent=0.2, img_size=(7,7), channels_df=None, score_map_same_sacale=True, channels=None):

    temp_map_1 = copy.deepcopy(score_map_1)
    temp_map_2 = copy.deepcopy(score_map_2)

    if tf.is_tensor(temp_map_1):
        temp_map_1 = temp_map_1.numpy()
    if tf.is_tensor(temp_map_2):
        temp_map_2 = temp_map_2.numpy()
    n_channels = img.shape[-1]

    # get the total variance corresponding to each channel
    score_channel_std_1 = []
    score_channel_std_2 = []
    for c in range(n_channels):
        score_channel_std_1.append(np.sum(temp_map_1[:,:,c]))
        score_channel_std_2.append(np.sum(temp_map_2[:,:,c]))
    total_var_1 = np.sum(score_channel_std_1)
    total_var_2 = np.sum(score_channel_std_2)

    # get the the top_percent pixels (make the rest 0)
    temp_map_1 = temp_map_1.reshape(-1)
    temp_map_2 = temp_map_2.reshape(-1)
    n_pixels = temp_map_1.shape[0]

    # If img mask given, then return the top MEASURED pixels
    if img_mask is None:
        cell_prop = 1
    else:
        cell_prop = np.sum(img_mask)*n_channels / n_pixels
    top_percent = cell_prop * top_percent
    n_top_pixels = int(top_percent * n_pixels)

    # get indices corresponding to the top pixels
    top_idxs_1 = temp_map_1.argsort()[-n_top_pixels:]
    top_idxs_2 = temp_map_2.argsort()[-n_top_pixels:]

    mask_1 = np.zeros(n_pixels)
    mask_1[top_idxs_1] = 1
    mask_1 = mask_1.astype(np.bool)
    mask_2 = np.zeros(n_pixels)
    mask_2[top_idxs_2] = 1
    mask_2 = mask_2.astype(np.bool)
    # make all non top pixels equal to zero
    temp_map_1[~mask_1] = 0
    temp_map_2[~mask_2] = 0
    # get min and max values for plot
    if score_map_same_sacale:
        vmin_1 = temp_map_1[mask_1].min()
        vmax_1 = temp_map_1[mask_1].max()
        vmin_2 = temp_map_2[mask_2].min()
        vmax_2 = temp_map_2[mask_2].max()
    else:
        vmin_1 = None
        vmax_1 = None
        vmin_2 = None
        vmax_2 = None

    temp_map_1 = temp_map_1.reshape(img.shape)
    temp_map_2 = temp_map_2.reshape(img.shape)

    if channels is None:
        channels = range(n_channels)
    n_channels = len(channels)

    plt.figure(figsize=(n_channels*img_size[0], 4*img_size[1]))

    for i, c in enumerate(channels, 1):
        channel_name = channels_df['name'][channels_df.channel_id == c].values[0]

        # Plot original image
        plt.subplot(5, n_channels, i)
        plot_cell(img=img[:,:,c], cmap='Blues', colorbar=True, title=channel_name)

        # Plot VarGrad_IG map 1 (importance map)
        channel_std_percen = round(100*score_channel_std_1[c] / total_var_1, 3)
        title='Score Map 1, Channel var: ' + str(channel_std_percen) + '%'
        plt.subplot(5, n_channels, n_channels+i)
        plot_cell(img=temp_map_1[:,:,c], cmap='Oranges', colorbar=True, title=title,
            vmin=vmin_1, vmax=vmax_1
            )

        # Plot VarGrad_IG map 2 (importance map)
        channel_std_percen = round(100*score_channel_std_2[c] / total_var_2, 3)
        title='Score Map 2, Channel var: ' + str(channel_std_percen) + '%'
        plt.subplot(5, n_channels, 2*n_channels+i)
        plot_cell(img=temp_map_2[:,:,c], cmap='Oranges', colorbar=True, title=title,
            vmin=vmin_2, vmax=vmax_2
            )

        # Plot image and importance map together
        plt.subplot(5, n_channels, 3*n_channels+i)
        plot_cell(img=temp_map_1[:,:,c], cmap='Oranges',
            vmin=vmin_1, vmax=vmax_1,
            colorbar=True
            )
        plot_cell(img=img[:,:,c], cmap='Blues', title='Cell and Score Map 1 overlap', alpha=0.4)

        # Plot image and importance map together
        plt.subplot(5, n_channels, 4*n_channels+i)
        plot_cell(img=temp_map_2[:,:,c], cmap='Oranges',
            vmin=vmin_2, vmax=vmax_2,
            colorbar=True
            )
        plot_cell(img=img[:,:,c], cmap='Blues', title='Cell and Score Map overlap', alpha=0.4)

    plt.show()

def plot_VarGrad_IG_2(img=None, img_mask=None, score_maps=None, top_percent=0.2, img_size=(7,7), channels_df=None, score_map_same_sacale=True, channels_2_plot=None, plot_overlap=False, plot_colorbar=True, plot_name=None):

    if channels_2_plot is None:
        channels_2_plot = range(img.shape[-1])

    n_channels = img.shape[-1]
    n_score_maps = len(score_maps.keys())
    n_pixels = np.prod(img.shape, axis=0)
    # Get the number of top pixels
    # If img mask given, then return the top MEASURED pixels
    if img_mask is None:
        cell_prop = 1
    else:
        cell_prop = np.sum(img_mask) * n_channels / n_pixels
    top_percent = cell_prop * top_percent
    n_top_pixels = int(top_percent * n_pixels)
    # Get number of rows and cols in the plot
    if plot_overlap:
        n_plot_rows = 1 + 2 * n_score_maps
    else:
        n_plot_rows = 1 + n_score_maps
    n_plot_columns = len(channels_2_plot)

    # set Plot size accordingly to the number of channels to plot and given maps
    fig = plt.figure(figsize=(n_plot_columns*img_size[0], n_plot_rows*img_size[1]))

    # Plot original image
    for i, c in enumerate(channels_2_plot, 1):
        channel_name = channels_df['name'][channels_df.channel_id == c].values[0]

        # Plot original image
        plt.subplot(n_plot_rows, n_plot_columns, i)
        plot_cell(img=img[:,:,c], cmap='Blues', colorbar=plot_colorbar, title=channel_name)

    # plot score maps
    row_count = 1
    for key in score_maps.keys():
        # Copy mask to avoid rewriting
        temp_map = copy.deepcopy(score_maps[key])
        if tf.is_tensor(temp_map):
            temp_map = temp_map.numpy()

        # get the total variance corresponding to each channel
        score_channel_std = []
        total_var = 0
        for c in range(n_channels):
            score_channel_std.append(np.sum(temp_map[:,:,c]))
        total_var = np.sum(score_channel_std)

        # get indices corresponding to the top pixels
        temp_map = temp_map.reshape(-1)
        top_idxs = temp_map.argsort()[-n_top_pixels:]
        # get the the top_percent pixels (make the rest 0)
        mask = np.zeros(n_pixels)
        mask[top_idxs] = 1
        mask = mask.astype(np.bool)
        temp_map[~mask] = 0
        # Return map to original original
        temp_map = temp_map.reshape(img.shape)

        # get min and max values for plot
        if score_map_same_sacale:
            vmin = temp_map[mask].min()
            vmax = temp_map[mask].max()
        else:
            vmin = None
            vmax = None

        # Plot score map
        for i, c in enumerate(channels_2_plot, 1):
            channel_name = channels_df['name'][channels_df.channel_id == c].values[0]

            # Get the variance (in the score map) corrsponding to this channel
            channel_stddev_percen = round(100*score_channel_std[c]/total_var,3)
            title=channel_name+', '+key+', ' + str(channel_stddev_percen) + '%'
            # plot
            plt.subplot(n_plot_rows, n_plot_columns, row_count*n_plot_columns + i)
            plot_cell(img=temp_map[:,:,c],
                      cmap='Oranges',
                      colorbar=plot_colorbar,
                      title=title,
                      vmin=vmin, vmax=vmax)
        row_count += 1

        # Plot overlap
        if plot_overlap:
            for i, c in enumerate(channels_2_plot, 1):
                channel_name = channels_df['name'][channels_df.channel_id == c].values[0]
                # Plot image and importance map together
                plt.subplot(n_plot_rows, n_plot_columns, row_count*n_plot_columns+i)
                plot_cell(img=temp_map[:,:,c],
                          cmap='Oranges',
                          vmin=vmin, vmax=vmax,
                          colorbar=False)
                title = channel_name+' and '+key+' Score Map'
                plot_cell(img=img[:,:,c], cmap='Blues', title=title, alpha=0.4)
            row_count += 1
    if plot_name is not None:
        fig.savefig(plot_name)
    plt.show()
