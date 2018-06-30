"""Encoder class that maps discrete symbol to continous vector"""

import numpy as np
import itertools
import cPickle as pickle
import tensorflow as tf
import util
import gumbel as gb


class Encoder(object):

  def __init__(self,
               K,
               D,
               d,
               outd,
               hparams,
               vocab_size,
               code_type="redundant",
               code_initializer=None,
               code_emb_initializer=None,
               create_code_logits=True,
               emb_baseline=False,
               pretrained_emb=None):
    """
    Args:
      K: a `int` number specifying the K-way choices of each dimension.
      D: a `int` number specifying the number of dims for the discrete code.
      d: a `int` number specifying the number of embedding dimension.
      outd: a `int` number specifying the number of output embedding dimension.
      code_type: a `string` specifies the type of code to use.
      code_initializer: a `string` filename to load previously saved one, or a
        numpy ndarray object.
    """
    self._vocab_size = vocab_size
    self._K = K
    self._D = D
    self._d = d
    self._outd = outd
    self._hparams = hparams
    self._code_type = code_type
    self._code_initializer = code_initializer
    self._code_emb_initializer = code_emb_initializer
    self._create_code_logits = create_code_logits
    self._emb_baseline = emb_baseline
    self._pretrained_emb = pretrained_emb

    self._code_emb = None
    with tf.variable_scope("code"):
      self._code_initialization()

  def _code_initialization(self):
    """Initialize the discrete code for symbols in vocab"""
    if self._code_type == "compact":
      init_code = np.array([
          code for code in itertools.product(*[
              range(self._K) for _ in range(self._D)])], dtype="int32")
      init_code = init_code[:self._vocab_size]
    elif self._code_type == "redundant":
      init_code = np.random.randint(0, self._K, self._vocab_size * self._D)
      init_code = init_code.reshape([-1, self._D])
    else:
      raise ValueError("Unknown code_type {}".format(self._code_type))
    init_code = tf.constant_initializer(init_code)

    preload_code = None
    if self._code_initializer is not None and len(self._code_initializer) > 0:
      if isinstance(self._code_initializer, str):
        with open(self._code_initializer) as fp:
          preload_code = pickle.load(fp)
        if len(preload_code.shape) != 2:
          raise ValueError("preload_code has to be in shape (vocab_size, D)")
        if str(preload_code.dtype).find("int") < 0:
          raise ValueError("preload_code has to be int dtype.")
        if preload_code.shape[1] != self._D or np.max(preload_code) != self._K - 1:
          tf.logging.warn("preload_code has D={},K={}, so transformed.".format(
              preload_code.shape[1], max(preload_code)))
        preload_code = preload_code[:, :self._D] % self._K
      elif isinstance(self._code_initializer, np.ndarray):
        preload_code = self._code_initializer
      else:
        raise ValueError("Unknown code_initializer {}".format(code_initializer))

    with tf.device("/gpu:0"):  # Could lead to much slower runtime.
      if self._hparams.ec_code_generator == "preassign":
        if preload_code is not None:
          print("Initialize the code from preload_code file {}".format(
              self._code_initializer))
          init_code = tf.constant_initializer(preload_code)
        self._code = tf.get_variable("code",
                                     [self._vocab_size, self._D],
                                     dtype=tf.int32,
                                     initializer=init_code,
                                     trainable=False)
      elif self._create_code_logits:
        if preload_code is not None:
          raise ValueError("Set the code_generator to preassign to use the "
                           "pretrained code.")
        if self._hparams.ec_code_generator in [
            "gumbel_softmax", "STE_argmax"]:
          self._code_logits = tf.get_variable("code_logits",
                                              [self._vocab_size,
                                               self._D, self._K],
                                              dtype=tf.float32,
                                              initializer=None)
        elif self._hparams.ec_code_generator in ["STE_threshold"]:
          if self._K != 2:
            raise ValueError("STE_threshold only works for K=2 binary coding.")
          self._code_logits = tf.get_variable("code_logits",
                                              [self._vocab_size, self._D],
                                              dtype=tf.float32,
                                              initializer=None)
        else:
          self._code_logits = None

      if self._emb_baseline:
        if self._pretrained_emb is None:
          initializer = None
          trainable = True
          shape = [self._vocab_size, self._outd]
        else:  # Use pretrained_emb purely as regularizer.
          initializer = self._pretrained_emb
          trainable = False
          shape = None
        self._embb = tf.get_variable(
            "embb", shape=shape, dtype=tf.float32,
            initializer=initializer, trainable=trainable)

  def encode(self, inputs, hparams):
    """Maps a batch of symbols to a batch of embeddings.

    This function first maps the symbols to codes by hash lookup, and then maps
    codes into embedding by combining code embeddings. This function is a
    combination of `symbol2code` & `embedding` contains learnable parameters.

    Args:
      inputs: a `Tensor` of shape (d1, d2, ..., dx).

    Returns:
      Embedding representations of (d1, d2, ..., dx, d) where d is the output
        embedding dimension.
    """
    codes = self.symbol2code(inputs)
    embs = self.embed(codes, hparams)

    return embs

  def embed(self,
            codes,
            code_embs=None,
            embsb=None,
            is_one_hot=False,
            hparams=None,
            is_training=False):
    """Maps a batch of codes to a batch of embeddings.

    This function maps codes into embedding by combining code embeddings.
    This function contains learnable parameters.

    Args:
      codes: a `Tensor` of shape (d1, d2, ..., dx, D), or (d1, d2, .., dx, D, K)
        when is_one_hot=True.
      code_embs: if not None, codes will be ignored.
      embsb: if given, use as embs baseline.
      is_one_hot: a `Bool` specifying whether or not the codes is in index
        format or one-hot format.

    Returns:
      Embedding representations of (d1, d2, ..., dx, d) where d is the output
        embedding dimension.
    """
    if hparams is None:
      hparams = self._hparams
    get_hparam = util.hparam_fn(hparams, prefix="ec")
    get_actv = util.filter_activation_fn()

    shared_coding = get_hparam("shared_coding")
    aggregator = get_hparam("aggregator")
    fnn_num_layers = get_actv(get_hparam("fnn_num_layers"))
    fnn_hidden_size = get_actv(get_hparam("fnn_hidden_size"))
    fnn_hidden_actv = get_actv(get_hparam("fnn_hidden_actv"))
    cnn_num_layers = get_hparam("cnn_num_layers")
    cnn_residual = get_hparam("cnn_residual")
    cnn_pooling = get_hparam("cnn_pooling")
    cnn_filters = get_hparam("cnn_filters")
    cnn_kernel_size = get_hparam("cnn_kernel_size")
    cnn_hidden_actv = get_actv(get_hparam("cnn_hidden_actv"))
    rnn_num_layers = get_hparam("rnn_num_layers")
    rnn_additive_pooling = get_hparam("rnn_additive_pooling")
    rnn_residual = get_hparam("rnn_residual")
    rnn_trainable_init_state = get_hparam("rnn_trainable_init_state")
    rnn_dropout = get_hparam("rnn_dropout")
    rnn_hidden_size = get_hparam("rnn_hidden_size")
    rnn_bidirection = get_hparam("rnn_bidirection")
    if self._emb_baseline:
      baseline_reg = get_hparam("emb_baseline_reg")
      baseline_dropout = get_hparam("emb_baseline_dropout")

    with tf.variable_scope("encode", reuse=not is_training):
      if code_embs is not None:
        embs = code_embs
      else:
        if is_one_hot:
          if shared_coding:
            code_embedding_size = [self._K]
          else:
            code_embedding_size = [self._D, self._K]
        else:
          if shared_coding:
            code_embedding_size = [self._K]
            codes_shifted = codes
          else:
            code_embedding_size = [self._D * self._K]
            shifts = tf.reshape(
                tf.range(self._D) * self._K,
                [1] * (len(codes.shape.as_list()) - 1) + [-1])
            # Move codes to same idx space for the ease of parameterization.
            codes_shifted = codes + shifts

        #with tf.device("/cpu:0"):  # Save memory; may lead to slower runtime.
        self._code_emb = tf.get_variable(
            "code_embedding",
            code_embedding_size + [self._d],
            initializer=self._code_emb_initializer,
              dtype=tf.float32)

        # Map codes to code embeddings, (d1, d2, ..., dx, D, d).
        if is_one_hot:
          if shared_coding:
            code_emb = tf.tile(
                tf.expand_dims(self._code_emb, 0),
                [self._D] + [1] * len(self._code_emb.shape.as_list()))
          else:
            code_emb = self._code_emb
          codes_shape_list = codes.shape.as_list()
          codes = tf.reshape(
              codes, (-1, codes_shape_list[-2], codes_shape_list[-1]))  # (-1, D, K)
          codes = tf.transpose(codes, [1, 0, 2])  # (D, -1, K)
          embs = tf.matmul(codes, code_emb)  # (D, -1, d)
          embs = tf.transpose(embs, [1, 0, 2])  # (-1, D, d)
          embs_shape = codes_shape_list[:]
          for i in range(len(embs_shape)):
            if embs_shape[i] is None:
              embs_shape[i] = -1
          embs_shape[-1] = self._d
          embs = tf.reshape(embs, embs_shape)
        else:
          embs = tf.nn.embedding_lookup(self._code_emb, codes_shifted)

      # Prepare the final embedding shape: (d1, ..., dx, outd)
      embs_shape = embs.shape.as_list()
      final_embs_shape = embs_shape[:-2] + [self._outd]
      final_embs_shape = util.replace_list_element(final_embs_shape, None, -1)

      # Map code embeddings to symbol embeddings, (d1, d2, ..., dx, d).
      if aggregator == "mean":
        embs = tf.reduce_mean(embs, -2)
        if self._outd != self._d:
          embs = tf.layers.batch_normalization(embs, training=is_training)
          embs = tf.layers.dense(embs, self._outd, use_bias=True)
          embs = tf.reshape(embs, final_embs_shape)
      elif aggregator == "mean_fnn":
        # First compute mean, then apply fnn.
        embs = tf.reduce_mean(embs, -2)
        embs = tf.reshape(embs, [-1, self._d])
        embs = tf.layers.batch_normalization(embs, training=is_training)
        embs = tf.layers.dense(embs,
                               fnn_hidden_size,
                               activation=fnn_hidden_actv)
        if fnn_num_layers == 0:
          raise ValueError("fnn_num_layers must be positive number.")
        for _ in range(fnn_num_layers - 1):
          embs = tf.layers.batch_normalization(embs, training=is_training)
          embs += tf.layers.dense(embs,
                                  fnn_hidden_size,
                                  activation=fnn_hidden_actv,
                                  use_bias=True)
        embs = tf.layers.batch_normalization(embs, training=is_training)
        embs = tf.layers.dense(embs, self._outd, use_bias=True)
        embs = tf.reshape(embs, final_embs_shape)
      elif aggregator == "fnn":
        # Directly apply fnn.
        embs = tf.reshape(embs, [-1, self._D * self._d])
        embs = tf.layers.dense(embs,
                               fnn_hidden_size,
                               activation=fnn_hidden_actv)
        if fnn_num_layers == 0:
          raise ValueError("fnn_num_layers must be positive number.")
        for _ in range(fnn_num_layers - 1):
          embs += tf.layers.dense(embs,
                                  fnn_hidden_size,
                                  activation=fnn_hidden_actv,
                                  use_bias=True)
        embs = tf.layers.dense(embs, self._outd, use_bias=True)
        embs = tf.reshape(embs, final_embs_shape)
      elif aggregator == "cnn":
        embs = tf.reshape(embs, [-1, self._D, self._d])
        for l in range(cnn_num_layers):
          embs_prev = embs
          embs = tf.layers.conv1d(embs,
                                  filters=cnn_filters,
                                  kernel_size=cnn_kernel_size,
                                  padding="same",
                                  data_format='channels_last',
                                  activation=cnn_hidden_actv)
          if cnn_residual:
            if l == 0 and self._d != cnn_filters:
              tf.logging.warn("Cannot apply CNN residual at the code embedding "
                              "level as the code embedding size is different "
                              "from the hidden size.")
            else:
              embs += embs_prev
        if cnn_pooling == "mean":
          embs = tf.reduce_mean(embs, 1)
        elif cnn_pooling == "max":
          embs = tf.reduce_max(embs, 1)
        else:
          raise ValueError("Unknown cnn_pooling {}".format(cnn_pooling))
        embs = tf.layers.dense(embs, self._outd, use_bias=False)
        embs = tf.reshape(embs, final_embs_shape)
      elif aggregator == "rnn":
        inputs = tf.reshape(embs, [-1, self._D, self._d])
        if is_training and rnn_dropout > 0:
          inputs = tf.nn.dropout(inputs, 1. - rnn_dropout)

        cell = tf.contrib.rnn.LSTMBlockCell(
            rnn_hidden_size, forget_bias=0.)
        #cell = tf.contrib.rnn.BasicLSTMCell(
        #    rnn_hidden_size, forget_bias=0.)
        #cell = tf.contrib.rnn.GRUBlockCell(
        #    rnn_hidden_size)
        #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        #    rnn_hidden_size, forget_bias=0.0)
        if is_training and rnn_dropout > 0:
          cell = tf.contrib.rnn.DropoutWrapper(
              cell, output_keep_prob=1. - rnn_dropout)
        if rnn_residual:
          cell = tf.contrib.rnn.ResidualWrapper(cell)  # before/after dropout?
        cell = tf.contrib.rnn.MultiRNNCell(
            [cell for _ in range(rnn_num_layers)], state_is_tuple=True)
        """ Using ResidualWrapper for layers except last one.
        cells = []
        for i in range(rnn_num_layers):
          if i == rnn_num_layers - 1:
            cells.append(cell)
          else:
            cells.append(tf.contrib.rnn.ResidualWrapper(cell))
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        """
        # Trainable initial states.
        if not rnn_trainable_init_state:
          initial_state = None
          # initial_state = cell.zero_state(tf.shape(inputs)[0], "float32")
        else:
          initial_state = []
          for i, state_size in enumerate(cell.state_size):
            if isinstance(state_size, tf.contrib.rnn.LSTMStateTuple):
              init_c = tf.get_variable("init_c_layer{}".format(i),
                                       [1, state_size.c],
                                       dtype=tf.float32,
                                       initializer=None,
                                       trainable=True)
              init_h = tf.get_variable("init_h_layer{}".format(i),
                                       [1, state_size.h],
                                       dtype=tf.float32,
                                       initializer=None,
                                       trainable=True)
              init_c = tf.tile(init_c, [tf.shape(inputs)[0], 1])
              init_h = tf.tile(init_h, [tf.shape(inputs)[0], 1])
              initial_state.append(tf.contrib.rnn.LSTMStateTuple(init_c, init_h))
            else:
              raise NotImplemented("TODO.")

        # Compute and generate (M, D, rnn_hidden_size).
        if not rnn_bidirection:
          inputs = tf.unstack(inputs, num=self._D, axis=1)
          outputs, state = tf.contrib.rnn.static_rnn(
              cell, inputs, initial_state=initial_state, dtype=tf.float32)
          if rnn_additive_pooling:
            outputs = [tf.expand_dims(score, 1) for score in outputs]
            outputs = tf.concat(outputs, 1)  # (M, D, rnn_hidden_size)
            outputs = tf.reduce_mean(outputs, 1)
          else:
            outputs = outputs[-1]
        else:
          inputs = tf.unstack(inputs, num=self._D, axis=1)
          outputs, _, _  = tf.contrib.rnn.static_bidirectional_rnn(
              cell, cell, inputs,
              initial_state_fw=None,  # TODO: add trainable init.
              initial_state_bw=None,
              dtype=tf.float32)
          if rnn_additive_pooling:
            outputs = [tf.expand_dims(score, 1) for score in outputs]
            outputs = tf.concat(outputs, 1)  # (M, D, rnn_hidden_size)
            outputs = tf.reduce_mean(outputs, 1)
          else:
            outputs = outputs[-1]

        #embs = outputs
        embs = tf.layers.dense(outputs, self._outd, use_bias=False)
        embs = tf.reshape(embs, final_embs_shape)
        #embs = tf.layers.batch_normalization(embs)
      else:
        raise ValueError("Unknown aggregator {}".format(aggregator))

      # BN before feeding it to the next layer. DEBUG?
      embs = tf.layers.batch_normalization(embs, training=is_training)

      # Adding full embedding baseline.
      baseline_reg2 = 0.  # TODO/hparams
      if embsb is not None and is_training:
        print("Emb_baseline enabled.")
        emb_baseline_reg_loss = self._l2_diff(embs, tf.stop_gradient(embsb))
        emb_baseline_revreg_loss = self._l2_diff(tf.stop_gradient(embs), embsb)
        tf.summary.scalar("emb_baseline_reg_loss", emb_baseline_reg_loss)
        tf.summary.scalar("emb_baseline_revreg_loss", emb_baseline_revreg_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             baseline_reg * emb_baseline_reg_loss +
                             baseline_reg2 * emb_baseline_revreg_loss)
        if self._pretrained_emb is None:
          if baseline_dropout == 0.:
            embs = (embs + embsb) / 2.
          else:
            embs_shape = self._get_shape(embs)
            embs_left = tf.reshape(embsb, [-1, embs_shape[-1]])
            embs_right = tf.reshape(embs, [-1, embs_shape[-1]])
            selector = baseline_dropout < tf.random_uniform(
                [np.prod(embs_shape[:-1])], dtype=tf.float32)
            embs = tf.where(selector, embs_left, embs_right)
            embs = tf.reshape(embs, embs_shape)
        else:
          # Otherwise purely use embsb as regularization.
          print("using pretrained_emb.")

    return embs

  def embed_transpose(self,
                      embs,
                      hparams=None,
                      is_training=False):
    """Maps a batch of embeddings to a batch of code logits.

    This function contains learnable parameters.

    Args:
      embs: a `Tensor` of (d1, d2, ..., dx, d) where d is the embedding dim.

    Returns:
      code_logits: a `Tensor` of shape (d1, d2, ..., dx, D, K).
    """
    if hparams is None:
      hparams = self._hparams
    get_hparam = util.hparam_fn(hparams, prefix="ec")
    get_actv = util.filter_activation_fn()

    num_layers = get_hparam("emb_transpose_layers")
    h_dim = get_hparam("emb_transpose_dim")
    h_actv = get_actv(get_hparam("emb_transpose_actv"))
    out_dim = self._D * self._K

    with tf.variable_scope("encode_transpose", reuse=not is_training):
      H = embs
      for l in range(num_layers):
        if l != num_layers - 1:
          H = h_actv(tf.layers.dense(H, h_dim))
        else:
          H = tf.layers.dense(H, out_dim)
          new_shape = self._get_shape(H)[:-1] + [self._D, self._K]
          H = tf.reshape(H, new_shape)
      code_logits = H
    return code_logits

  def _l2_diff(self, X, Y):
    return tf.reduce_mean((X - Y)**2)

  def _mmd(self, X, Y, sigma):
    """Return mmd loss given parameterized X and fixed target Y.
    Only final dimension is feature dimension.
    """
    print("happy mmd! sigma %f"%sigma)
    X_shape = self._get_shape(X)
    Y_shape = self._get_shape(Y)
    X = tf.reshape(X, [-1, X_shape[-1]])
    Y = tf.reshape(Y, [-1, Y_shape[-1]])

    def _kernel(x, y, sigma):
      x_sqr = tf.reduce_sum(x**2, -1, keep_dims=True)
      y_sqr = tf.reduce_sum(y**2, -1, keep_dims=True)
      xy_l2 = x_sqr - 2.*tf.matmul(x, y, transpose_b=True) + tf.transpose(y_sqr)
      # xy_l2 = tf.layers.batch_normalization(xy_l2, training=True)  # DEBUG
      return tf.exp(- 0.5 * xy_l2 / sigma)

    # jj = tf.reshape(_kernel(X, X, sigma), [-1])
    # X = tf.Print(X, [tf.nn.moments(jj, 0)]) DEBUG
    loss_mmd = tf.reduce_mean(_kernel(X, X, sigma))
    loss_mmd += -2. * tf.reduce_mean(_kernel(X, Y, sigma))  # + const from Y
    loss_mmd += tf.reduce_mean(_kernel(Y, Y, sigma))
    return tf.sqrt(loss_mmd)

  def symbol2code(self,
                  inputs,
                  logits=None,
                  hparams=None,
                  reuse=None,
                  output_embb=False,
                  is_training=False,
                  output_logits=False,
                  logits_bn_overwrite=None):
    """Maps a batch of symbols into a batch of codes.

    Args:
      inputs: a id `Tensor` of shape (d1, d2, ..., dx).
      logits: a logits `Tensor` of shape (d1, d2, ..., dx, K, D), if given,
        inputs will be ignored, otherwise it is indexed from self._code_logits
        using inputs.

    Returns:
      According to the generator, one of the following code would be returned:
        A `Tensor` of (d1, d2, ..., dx, D) where each symbol is replaced by a
          D-dimensional code.
        A `Tensor` of (d1, d2, ..., dx, D, K) where each symbol is replaced by a
          D-dimensional code in one-hot embedding format.
      code_embs: only not None for vq method, others will be created in embed()
    """
    if hparams is None:
      hparams = self._hparams
    get_hparam = util.hparam_fn(hparams, prefix="ec")
    logits_bn = get_hparam("logits_bn")
    code_generator = get_hparam("code_generator")
    code_dropout = get_hparam("code_dropout")
    hard = get_hparam("hard_code_output")
    STE_softmax = get_hparam("STE_softmax_transform")
    entropy_reg = get_hparam("entropy_reg")
    decay_method = get_hparam("temperature_decay_method")
    decay_steps = get_hparam("temperature_decay_steps")
    decay_rate = get_hparam("temperature_decay_rate")
    t_init = get_hparam("temperature_init")
    t_low = get_hparam("temperature_lower_bound")
    if decay_method == "none":
      temperature = 1.0
    else:
      temperature = (decay_method, decay_steps, decay_rate, t_init, t_low)

    if reuse is None:
      try:
        self._symbol2code_reuse
        reuse = True
      except:
        self._symbol2code_reuse = True
        reuse = False

    with tf.variable_scope("symbol2code", reuse=reuse):
      code_embs = None
      # One-hot encoding output for all code_generators except preassign.
      if code_generator == "preassign":
        codes = tf.nn.embedding_lookup(self._code, inputs)
      else:
        if logits is None:
          logits = tf.nn.embedding_lookup(self._code_logits, inputs)
        if logits_bn_overwrite is not None:
          logits_bn = logits_bn_overwrite
        if logits_bn > 0.:
          if decay_method == "none":
            center, scale = True, True
          else:
            center, scale = False, False
          logits = logits_bn * tf.layers.batch_normalization(
              logits, training=is_training, center=center, scale=scale)
        if code_generator == "gumbel_softmax":
          codes, codes_soft, _ = gb.gumbel_softmax(
              logits,
              temperature=temperature,
              entropy_reg=entropy_reg,
              random=is_training,  # Use argmax for test.
              straight_through=True,
              is_training=is_training)
          codes = codes if hard or (not is_training) else codes_soft  # Use hard for test.
        elif code_generator == "STE_argmax":
          codes = gb.straight_through(
              logits,
              thresholding=False,
              softmax=STE_softmax,
              hard=hard or (not is_training),  # Use hard for test.
              temperature=temperature,
              entropy_reg=entropy_reg,
              is_training=is_training)
        elif code_generator == "STE_threshold":
          codes = gb.straight_through(
              logits,
              thresholding=True,
              softmax=STE_softmax,
              hard=hard or (not is_training),  # Use hard for test.
              temperature=temperature,
              entropy_reg=entropy_reg,
              is_training=is_training)
        else:
          raise ValueError("Unknown code_generator {}".format(code_generator))

    if code_dropout > 0. and is_training:
      # Randomly dropout for each examples and each of D dimension of the code.
      codes_shape = codes.shape.as_list()
      noise_shape = [1] * len(codes_shape)
      noise_shape[0], noise_shape[-2] = tf.shape(codes)[0], self._D
      codes = tf.nn.dropout(
          codes, keep_prob=1. - code_dropout, noise_shape=noise_shape)

    if output_embb:
      if not self._emb_baseline:
        raise ValueError("output_embb can only be True when emb_baseline=True")
      embs = tf.nn.embedding_lookup(self._embb, inputs)
      if output_logits:
        return codes, code_embs, embs, logits
      else:
        return codes, code_embs, embs
    else:
      if output_logits:
        return codes, code_embs, logits
      else:
        return codes, code_embs

  def code2symbol(codes):
    """Maps a batch of codes into a batch of symbols.

    Args:
      codes: a `Tensor` of (batch_size, D)

    Returns:
      A `Tensor` of (batch_size, 1) where each code is replaced by a symbol.
    """
    raise NotImplemented("TODO")

  def _get_shape(self, T):
    """Return tensor shape, as much as integer."""
    T_shape = T.shape.as_list()
    if not isinstance(T_shape[0], int):
      T_shape[0] = tf.shape(T)[0]
    return T_shape

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def code(self):
    """Returns code table for all symbols."""
    return self._code

  @property
  def code_embedding(self):
    """Returns code embedding table."""
    if self._code_emb is None:
      raise ValueError("encode() should run first.")
    return self._code_emb

  @property
  def code_logits(self):
    return self._code_logits
