import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_cell(img, cmap='PiYG', title='', vmin=0, vmax=1, colorbar=False, alpha=1):
    fig = plt.imshow(img,
               cmap=getattr(plt.cm, cmap),
               vmin=vmin, vmax=vmax,
               aspect='equal',
               alpha=alpha
               )
    plt.title(title)

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

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients

@tf.function
def integrated_gradients(baseline, image, model, m_steps=50, batch_size=32):
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
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

def get_VarGrad(img=None, img_mask=None, baseline='black', model=None, n_images=15):

    if baseline == 'black':
        bl = tf.zeros(shape=img.shape)
    else:
        raise NotImplementedError('{} baseline method not implemented'.format(baseline))

    for i in range(n_images):
        print('Computing IG ', i+1)
        rnorm_img = np.random.normal(loc=0, scale=1, size=img.shape).astype(np.float32)

        # If image mask is given, then only add noice into measured pixels
        if img_mask is not None:
            rnorm_img[~img_mask] = 0

        ig_temp = integrated_gradients(baseline=bl,
                                       image=img + rnorm_img,
                                       model=model,
                                       m_steps=240)

        ig_temp = tf.expand_dims(ig_temp, axis=0)

        if i == 0:
            ig_sample = ig_temp
        else:
            ig_sample = tf.concat((ig_sample, ig_temp), axis=0)

    return tf.math.reduce_std(ig_sample, axis=0)

def plot_VarGrad_IG(img=None, img_mask=None, importance_map=None, top_percent=0.2, img_size=(7,7), channels_df=None):

    temp_score = importance_map.numpy()
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
    vmin = temp_score[mask].min()
    vmax = temp_score[mask].max()

    temp_score = temp_score.reshape(original_shape)

    plt.figure(figsize=(n_channels*img_size[0], 2*img_size[1]))

    # plot sorting by channel var percent
    for i, c in enumerate(np.argsort(-np.array(score_channel_std)), 1):
        channel_name = channels_df['name'][channels_df.channel_id == c].values[0]

        # Plot original image
        plt.subplot(3, n_channels, i)
        plot_cell(img=img[:,:,c], cmap='Blues', colorbar=True, title=channel_name)

        # Plot VarGrad_IG map (importance map)
        channel_std_percen = round(100*score_channel_std[c] / total_var, 3)
        title='Total channel var: ' + str(channel_std_percen) + '%'
        plt.subplot(3, n_channels, n_channels+i)
        plot_cell(img=temp_score[:,:,c], cmap='Oranges', colorbar=True, title=title,
            vmin=None, vmax=None
            #vmin=vmin, vmax=vmax
            )

        # Plot image and importance map together
        plt.subplot(3, n_channels, 2*n_channels+i)
        plot_cell(img=temp_score[:,:,c], cmap='Oranges', title=channel_name,
            vmin=None, vmax=None,
            #vmin=vmin, vmax=vmax,
            colorbar=True
            )
        plot_cell(img=img[:,:,c], cmap='Blues', title=channel_name, alpha=0.4)

    plt.show()
