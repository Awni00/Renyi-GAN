import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def renyi_crossentropy(target, output, alpha, epsilon=0.0001):
    '''renyi crossentropy function'''
    real_loss = tf.math.reduce_mean(
        tf.math.pow(target, (alpha - 1) * tf.ones_like(target))
    )
    real_loss = 1.0 / (alpha - 1) * tf.math.log(real_loss + epsilon) + tf.math.log(2.0)

    f = tf.math.reduce_mean(tf.math.pow(1 - output, (alpha - 1) * tf.ones_like(output)))
    gen_loss = 1.0 / (alpha - 1) * tf.math.log(f + epsilon) + tf.math.log(2.0)

    loss = -real_loss - gen_loss

    return loss


def create_gen_loss(alpha, epsilon=0.0001, lamda=0):
    """
    create generator loss

    Args:
        alpha (float): alpha value for (renyi) generator loss. (if alpha=1, standard crossentropy is used)
        epsilon (float, optional): for numerical stability. Defaults to 0.0001.
        lamda (float, optional): regularization. Defaults to 0.

    Returns:
        function: generator loss function
    """

    def standard_generator_loss(disc_generated_output, gen_output, target):

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (lamda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def renyi_generator_loss(disc_generated_output, gen_output, target):
        gan_loss = renyi_crossentropy(
            tf.ones_like(disc_generated_output),
            disc_generated_output,
            alpha=alpha,
            epsilon=epsilon,
        )

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (lamda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    if alpha==1:
        return standard_generator_loss
    else:
        return renyi_generator_loss

