"""Gumbel Softmax utilities
"""

import tensorflow as tf
import util
safer_log = util.safer_log
Beta = tf.distributions.Beta


def _get_weight_schedule(weight,
                         name,
                         clipping=None,
                         is_training=False):
  """Generate weight schedule.

  Args:
    weight: a `Tensor` for controlling weight. Or a list specifying
      (decay_method, decay_steps, decay_rate, init, lower_bound).
    clipping: a pair specifying the lower and upper bounds of the weight.

  Returns:
    identity if the input weight is `Tensor`, else return a `Tensor`
      weight that has automatic schedule.
  """
  if isinstance(weight, list) or isinstance(weight, tuple):
    method, decay_steps, decay_rate, init, lower_bound = weight
    if clipping is None:
      clipping = (lower_bound, init)
    if method == "exp":
      decayer = tf.train.exponential_decay
    elif method == "inv_time":
      decayer = tf.train.inverse_time_decay
    else:
      raise ValueError("Unknown weight_decay_method {}".format(method))
    weight = tf.get_variable(
        name, initializer=init, dtype=tf.float32, trainable=False)
    weight = decayer(
        weight, tf.train.get_or_create_global_step(),
        decay_steps, decay_rate, staircase=True)
  elif not isinstance(weight, tf.Tensor):
    raise ValueError("weight has to be either a tf.Tensor or a list of "
                     "(method, decay_steps, decay_rate). Now it is {}".format(
                      weight))
  if clipping is not None:
    weight = tf.clip_by_value(weight, clipping[0], clipping[1])
  if is_training:
    tf.summary.scalar(name, weight)
  return weight


def sample_gumbel(shape):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape, minval=0, maxval=1)
  return -safer_log(-safer_log(U))


def sample_z(log_theta, random=True):
  """z = log_theta + gumbel_variable"""
  if random:
    z = log_theta + sample_gumbel(tf.shape(log_theta))
  else:
    z = log_theta
  return z


def gumbel_softmax(logits,
                   temperature=1.,
                   entropy_reg=0.,
                   random=True,
                   straight_through=False,
                   logits_are_probas=False,
                   return_raw_z=False,
                   is_training=False):
  """Sample from the Gumbel-Softmax distribution.

  Args:
    logits: unnormalized log-probs of shape (batch_size, D, n_class).
    temperature: a `Tensor` for controlling temperature. Or a list specifying
      (decay_method, decay_steps, decay_rate).
    entropy_reg: a `Tensor` for controlling regularization weight for entropy.
      Or a list specifying (decay_method, decay_steps, decay_rate). When using
      entropy_reg, you need manually to add the regularization for minimizer!
    random: if False, degenerate to K-way straight_through estimator
    straight_through: whether or not to set straight-through the hard sample b.

  Returns:
    1) hard samples b = argmax(z),
    2) soft samples softmax(z, tau) of the same shape,
    3) soft samples softmax(zb, tau) conditioned on b.Would be None if
      random=False.
  """
  if isinstance(temperature, list) or isinstance(temperature, tuple):
    temperature = _get_weight_schedule(
        temperature, "temperature", is_training=is_training)
  if logits_are_probas:
    theta = logits
  else:
    theta = tf.nn.softmax(logits)
  log_theta = safer_log(theta)
  z = sample_z(log_theta, random=random)
  b = tf.cast(tf.one_hot(tf.argmax(z, -1), tf.shape(z)[-1]), z.dtype)
  z_softmax = tf.nn.softmax(tf.div(z, temperature))
  if is_training:
    gumbel_entropy = -tf.reduce_mean(tf.reduce_sum(theta * log_theta, -1))
    tf.summary.scalar("gumbel_entropy", gumbel_entropy)

  if random:  # Sample zb conditioned b.
    """Implementation 1
    theta_b = tf.reduce_max(theta, -1, keep_dims=True)
    vb = Beta(1. + tf.div(1. - theta_b, theta_b), 1.).sample()
    vb_weighted = tf.pow(vb, tf.div(theta, theta_b))
    vi_weighted = tf.pow(tf.random_uniform(tf.shape(theta), minval=0, maxval=1),
                         1. - b)
    v = vi_weighted * vb_weighted
    zb = log_theta + tf.stop_gradient(-safer_log(-safer_log(v)))
    #zb = log_theta - safer_log(-safer_log(v))  # TODO/DEBUG: TEST grad.
    """
    """Implementation 2 from https://github.com/Bonnevie/rebar/blob/2f526300eed123af64f6df601fdfaea3f5fd6dd5/relaxflow/reparam.py#L227
    """
    def truncated_gumbel(gumbel, truncation):
        return -safer_log(tf.exp(-gumbel) + tf.exp(-truncation))
    u = sample_gumbel(tf.shape(log_theta))
    topgumbels = u + tf.reduce_logsumexp(log_theta, axis=-1, keep_dims=True)
    topgumbel = tf.reduce_sum(b * topgumbels, axis=-1, keep_dims=True)
    truncgumbel = truncated_gumbel(u + log_theta, topgumbel)
    zb = (1. - b) * truncgumbel + b * topgumbels

    zb_softmax = tf.nn.softmax(tf.div(zb, temperature))
  else:
    zb_softmax = zb = None

  if straight_through:
    b = tf.stop_gradient(b - z_softmax) + z_softmax

  if entropy_reg != 0. and is_training:
    if isinstance(entropy_reg, list) or isinstance(entropy_reg, tuple):
      entropy_reg = _get_weight_schedule(
          entropy_reg, "entropy_reg", is_training=is_training)
    entropy_reg_loss = gumbel_entropy * entropy_reg
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, entropy_reg_loss)

  if return_raw_z:
    return b, z_softmax, zb_softmax, z, zb
  else:
    return b, z_softmax, zb_softmax


def gumbel_binary(logits,
                  temperature=1.,
                  entropy_reg=0.,
                  random=True,
                  straight_through=False,
                  logits_are_probas=False,
                  return_raw_z=False,
                  is_training=False):
  """Gumbel_softmax but when logits are the scores for positive class.
  """
  if isinstance(temperature, list) or isinstance(temperature, tuple):
    temperature = _get_weight_schedule(
        temperature, "temperature", is_training=is_training)
  sigmoid = lambda x, tau: 1. / (1. + tf.exp(-tf.div(x, tau)))
  logit = lambda x: safer_log(tf.div(x, 1. - x + util.eps_tiny))
  if logits_are_probas:
    theta = logits
  else:
    theta = sigmoid(logits, 1.)
  logit_theta = logit(theta)
  u = tf.random_uniform(tf.shape(theta), minval=0, maxval=1)
  logit_u = logit(u)
  if random:
    z = logit_theta + logit_u
    b = tf.cast(z >= 0., tf.float32)
    z_sigmoid = sigmoid(z, temperature)
    v = tf.random_uniform(tf.shape(theta), minval=0, maxval=1)
    vp = (1. - b) * v * (1. - theta) + b * (v * theta + (1. - theta))
    logit_v = logit(vp)
    zb = logit_theta + logit_v
    zb_sigmoid = sigmoid(zb, temperature)
  else:
    z = logit_theta
    z_sigmoid = sigmoid(z, temperature)
    b = tf.cast(z >= 0., tf.float32)
    zb_sigmoid = zb = None

  if straight_through:
    b = tf.stop_gradient(b - z_sigmoid) + z_sigmoid

  if entropy_reg != 0 and is_training:
    raise NotImplemented("Right now entropy_reg is not supported for binary")

  if return_raw_z:
    return b, z_sigmoid, zb_sigmoid, z, zb
  else:
    return b, z_sigmoid, zb_sigmoid


def straight_through(logits,
                     thresholding=False,
                     softmax=False,
                     hard=True,
                     temperature=1.,
                     entropy_reg=0.,
                     is_training=False):
  """Straight-through estimator for discrete node.

  This function takes logits and turn them into one-hot activations, which
  corresponds to the discrete activations/actions.

  Args:
    logits: a `Tensor` of shape (batch_size, D, n_class).
    thresholding: whether to use binary activation and thresholding, or use
      K-way softmax activation and argmax.
    softmax: whether to use softmax/sigmoid to transform the logits.
    hard: whether output a hard or soft one-hot encoding vector.
    temperature: a `Tensor` for controlling temperature. Or a list specifying
      (decay_method, decay_steps, decay_rate).
    entropy_reg: a `Tensor` for controlling regularization weight for entropy.
      Or a list specifying (decay_method, decay_steps, decay_rate). When using
      entropy_reg, you need manually to add the regularization for minimizer!
    is_training: a `Bool` specifying whether or not to do summarization.

  Returns:
    One-hot activations of shape (batch_size, D, n_class).
  """
  original_logits = logits
  if isinstance(temperature, list) or isinstance(temperature, tuple):
    temperature = _get_weight_schedule(
        temperature, "temperature", is_training=is_training)
    logits = tf.div(logits, temperature)

  if thresholding:
    # Using binary activation and thresholding.
    if softmax:
      y = tf.nn.sigmoid(logits)
      level = 0.5
      reverse_v = 1.
    else:
      # y = tf.clip_by_value(logits, -1, 1)
      y = logits
      level = 0.
      reverse_v = 0.
    y_hard = tf.cast(tf.one_hot(tf.cast(y > level, tf.int32), 2), y.dtype)
    y = tf.expand_dims(y, -1)
    y = tf.concat([reverse_v - y, y], -1)
  else:
    # Using K-way (softmax) activation and argmax.
    if softmax:
      y = tf.nn.softmax(logits)
      if is_training:
        gumbel_entropy = -tf.reduce_mean(tf.reduce_sum(y * safer_log(y), -1))
        tf.summary.scalar("gumbel_entropy", gumbel_entropy)
    else:
      # y = tf.clip_by_value(logits, -1, 1)
      y = logits
    y_hard = tf.cast(
        tf.one_hot(tf.argmax(original_logits, -1), tf.shape(y)[-1]),
        y.dtype)

  if hard:
    y = tf.stop_gradient(y_hard - y) + y

  if entropy_reg != 0. and is_training:
    if thresholding:
      raise NotImplemented("Right now entropy_reg is not supported for binary")
    if isinstance(entropy_reg, list) or isinstance(entropy_reg, tuple):
      entropy_reg = _get_weight_schedule(
          entropy_reg, "entropy_reg", is_training=is_training)
    entropy_reg_loss = gumbel_entropy * entropy_reg
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, entropy_reg_loss)

  return y
