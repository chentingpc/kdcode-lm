# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Modified based on https://github.com/tensorflow/models/tree/r1.4.0/tutorials/rnn/ptb.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import inspect

import cPickle as pickle
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf

import reader
import util
from encoder import Encoder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable when using tf.Print

flags = tf.flags
logging = tf.logging

# Define basics.
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("dataset", "ptb",
                    "Supported dataset: ptb, text8, 1billion.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_integer("save_model_secs", 0,
                    "Setting it zero to avoid saving the checkpoint.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats.")
flags.DEFINE_bool("use_recoding", False,
                  "Whether or not to apply discrete recoding technique.")
flags.DEFINE_integer("max_max_epoch", None,
                     "The total number of epochs for training.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_bool("rnn_residual", None,
                  "Whether or not to use residual connection in LM RNNs.")
flags.DEFINE_bool("shared_embedding", None,
                  "Whether or not to share the encoder embedding for decoder.")
flags.DEFINE_float("max_grad_norm", None,
                   "the maximum permissible norm of the gradient.")
flags.DEFINE_string("optimizer", "lazy_adam",
                   "The predefined optimizer solution.")
flags.DEFINE_float("learning_rate", None,
                   "Learning rate.")
flags.DEFINE_float("weight_decay", None,
                   "weight decay, or l2 penalty for all trainable parameters.")
flags.DEFINE_float("emb_lowrank_dim", 0, "")
flags.DEFINE_bool("test_summary", False, "")

# Define encoder/decoder related.
flags.DEFINE_integer("D", None, "D-dimensional code.")
flags.DEFINE_integer("K", None, "K-way code.")
flags.DEFINE_string("code_type", None, "Type of the code.")
flags.DEFINE_string("code_save_filename", None, "")
flags.DEFINE_string("code_load_filename", None, "")
flags.DEFINE_string("emb_save_filename", None, "")
flags.DEFINE_string("emb_load_filename", None, "")
flags.DEFINE_string("ec_code_generator", None, "")
flags.DEFINE_bool("ec_emb_baseline", None, "")
flags.DEFINE_float("ec_emb_baseline_reg", None, "")
flags.DEFINE_float("ec_emb_baseline_reg2", None, "")
flags.DEFINE_float("ec_emb_baseline_reg3", None, "")
flags.DEFINE_float("ec_emb_baseline_dropout", None, "")
flags.DEFINE_float("ec_logits_bn", None, "")
flags.DEFINE_float("ec_entropy_reg", None, "")
flags.DEFINE_string("ec_temperature_decay_method", None, "")
flags.DEFINE_integer("ec_temperature_decay_steps", None, "")
flags.DEFINE_float("ec_temperature_decay_rate", None, "")
flags.DEFINE_float("ec_temperature_init", None, "")
flags.DEFINE_float("ec_temperature_lower_bound", None, "")
flags.DEFINE_bool("ec_shared_coding", None, "")
flags.DEFINE_float("ec_code_dropout", None, "")
flags.DEFINE_bool("ec_STE_softmax_transform", None, "")
flags.DEFINE_bool("ec_hard_code_output", None, "")
flags.DEFINE_integer("ec_code_emb_dim", None, "")
flags.DEFINE_string("ec_aggregator", None, "")
flags.DEFINE_integer("ec_fnn_num_layers", None, "")
flags.DEFINE_integer("ec_fnn_hidden_size", None, "")
flags.DEFINE_string("ec_fnn_hidden_actv", None, "")
flags.DEFINE_integer("ec_cnn_num_layers", None, "")
flags.DEFINE_bool("ec_cnn_residual", None, "")
flags.DEFINE_string("ec_cnn_pooling", None, "")
flags.DEFINE_integer("ec_cnn_filters", None, "")
flags.DEFINE_integer("ec_cnn_kernel_size", None, "")
flags.DEFINE_string("ec_cnn_hidden_actv", None, "")
flags.DEFINE_integer("ec_rnn_num_layers", None, "")
flags.DEFINE_integer("ec_rnn_hidden_size", None, "")
flags.DEFINE_bool("ec_rnn_additive_pooling", None, "")
flags.DEFINE_bool("ec_rnn_residual", None, "")
flags.DEFINE_float("ec_rnn_dropout", None, "")
flags.DEFINE_bool("ec_rnn_trainable_init_state", None, "")
flags.DEFINE_bool("ec_rnn_bidirection", None, "")
flags.DEFINE_bool("ec_emb_autoencoding", None, "")
flags.DEFINE_integer("ec_emb_transpose_layers", None, "")
flags.DEFINE_integer("ec_emb_transpose_dim", None, "")
flags.DEFINE_string("ec_emb_transpose_actv", None, "")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

TRAIN_MODE = "train"
VALID_MODE = "valid"
TEST_MODE = "test"

id2word = None  # sloppy
vocab_size = None
last_cs = None


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def print_at_beginning(hparams):
  global vocab_size
  print("vocab_size={}, K={}, D={}, code_type={}".format(
      vocab_size, hparams.K, hparams.D, hparams.code_type))
  print("Number of trainable params:    {}".format(
      util.get_parameter_count(
          excludings=["code_logits", "embb", "symbol2code"])))


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, vocab_size, pretrained_emb):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self._config = config
    self._pretrained_emb = pretrained_emb
    size = config.hidden_size

    inputs = None
    if FLAGS.use_recoding:
      if config.ec_code_generator == "preassign":
        is_one_hot = False
      else:
        is_one_hot = True
      encoder = Encoder(K=config.K,
                        D=config.D,
                        d=config.ec_code_emb_dim,
                        outd=size,
                        hparams=config,
                        vocab_size=vocab_size,
                        code_type=config.code_type,
                        code_initializer=FLAGS.code_load_filename,
                        emb_baseline=config.ec_emb_baseline,
                        code_emb_initializer=None,
                        create_code_logits=True,
                        pretrained_emb=pretrained_emb)
      if not config.disable_encoding:
        if config.ec_emb_baseline:
          code_logits = None
          if config.ec_emb_autoencoding and is_training:
            print("emb_baseline with auto-encoding enabled.")
            embsb = tf.nn.embedding_lookup(pretrained_emb, input_.input_data)
            embsb = tf.stop_gradient(embsb)
            code_logits_b = encoder.embed_transpose(
                embsb, is_training=is_training)  # (bs, steps, D, K)
            if config.ec_temperature_decay_method == "none":
              center, scale = True, True
            else:
              center, scale = False, False
            with tf.variable_scope("autoencoding_code_logits_bn", reuse=False):
              if config.ec_logits_bn > 0:
                code_logits_b = config.ec_logits_bn * tf.layers.batch_normalization(
                    code_logits_b, training=is_training, center=center, scale=scale)
            codes_m, _, code_logits = encoder.symbol2code(
                input_.input_data, is_training=is_training, output_logits=True)
            codes_b, _ = encoder.symbol2code(  # has to be after codes_m for bn issue.
                input_.input_data, logits=code_logits_b,
                logits_bn_overwrite=0, is_training=is_training)
            codes_concat = tf.concat([codes_b, codes_m], 0)
            embs = encoder.embed(  # Shared code->emb function.
                codes_concat, is_one_hot=is_one_hot, is_training=is_training)
            embsb_reconst, embs_m = tf.split(embs, 2, axis=0)
            inputs = embs_m  # (batch_size, steps, emb_dim)
            # Add regularization.
            regl_embs = tf.reduce_mean((embsb - embs_m)**2)
            regl_reconst = tf.reduce_mean((embsb - embsb_reconst)**2)
            regl_logits = tf.reduce_mean((code_logits_b - code_logits)**2)
            if is_training:
              tf.summary.scalar("regl_embs", regl_embs)
              tf.summary.scalar("regl_reconst", regl_reconst)
              tf.summary.scalar("regl_logits", regl_logits)
            emb_baseline_loss = (regl_embs * config.ec_emb_baseline_reg +
                                 regl_reconst * config.ec_emb_baseline_reg2 +
                                 regl_logits * config.ec_emb_baseline_reg3)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 emb_baseline_loss)
          else:
            input_codes, code_embs, embsb = encoder.symbol2code(
                input_.input_data, logits=code_logits,
                is_training=is_training, output_embb=True)
        else:  # no emb baseline.
          if config.ec_emb_autoencoding:
            raise ValueError("ec_emb_autoencoding should be False when "
                             "ec_emb_baseline is False. Please check!")
          embsb = None
          input_codes, code_embs = encoder.symbol2code(
              input_.input_data, is_training=is_training)

        if inputs is None:  # will not enter here if ec_emb_autoencoding=True.
          inputs = encoder.embed(input_codes,
                                 code_embs=code_embs,
                                 embsb=embsb,
                                 is_one_hot=is_one_hot,
                                 is_training=is_training)

        self.query_codes, _code_embs = encoder.symbol2code(
            tf.range(vocab_size), is_training=False)
        self.query_input_emb = encoder.embed(
            self.query_codes, _code_embs,
            is_one_hot=is_one_hot, is_training=False)

    if inputs is None:
      with tf.device("/gpu:0"):
        if config.emb_lowrank_dim == 0:
          embedding = tf.get_variable(
              "embedding", [vocab_size, size], dtype=data_type())
          if pretrained_emb is not None:
            self.using_pretrained_embs_on_run_op = (
                reload_embedding_after_checkpoint_recover(pretrained_emb))
          inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
          self.query_codes = None
          self.query_input_emb = embedding
        else:
          _dim = config.emb_lowrank_dim
          if _dim < 1.:  # Represents ratio of low-rank parameters to full one.
            _dim = int(_dim * vocab_size / (vocab_size + size) * size)
          embedding_p = tf.get_variable(
              "embedding_p", [vocab_size, _dim], dtype=data_type())
          embedding_q = tf.get_variable(
              "embedding_q", [_dim, size], dtype=data_type())
          inputs = tf.nn.embedding_lookup(embedding_p, input_.input_data)
          inputs = tf.tensordot(inputs, embedding_q, [[-1], [0]])
          self.query_codes = None
          self.query_input_emb = embedding_p
    if config.ec_code_generator == "preassign":
        self.query_codes = None  # save memory and space.

    targets = input_.targets
    self.batch_size_final = self.batch_size
    outputs, state = self._build_rnn_graph(inputs, config, is_training)

    if config.shared_embedding:
      softmax_w = tf.transpose(embedding)
    else:
      softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
    self.query_output_emb = tf.transpose(softmax_w)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits,
                        [self.batch_size_final, self.num_steps, vocab_size])
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([self.batch_size_final, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=False)  # (batch_size, num_steps)

    # Update the cost
    self._nll = tf.reduce_sum(tf.reduce_mean(loss, 0))
    self._cost = self._nll
    self._final_state = state

    if not is_training:
      return


    # Add regularization.
    self._cost += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add weight decay.
    if config.weight_decay != 0.:
      for tvar in tf.trainable_variables():
        self._cost += config.weight_decay * 0.5 * tf.reduce_sum(tf.square(tvar))

    self._lr = tf.Variable(0.0, trainable=False)
    non_pg_tvars = tf.trainable_variables()
    if config.optimizer == "mixed":
      if config.ec_code_generator == "preassign":
        raise ValueError("Shouldn't use mixed when using preassign.")
      mixed_encode = False
      pg_tvars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="Model/code/code_logits")
      pg_tvars += tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="Model/symbol2code")
      if mixed_encode:
        pg_tvars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="Model/encode")
      tvars, non_pg_tvars = non_pg_tvars, []
      for tvar in tvars:
        if tvar.name.startswith("Model/code/code_logits") or (
            tvar.name.startswith("Model/symbol2code")):
          continue
        if mixed_encode and tvar.name.startswith("Model/encode"):
          continue
        non_pg_tvars.append(tvar)
      if len(tvars) == len(non_pg_tvars) or (
          len(pg_tvars) + len(non_pg_tvars) != len(tvars)):
        raise ValueError("Check pg_tvars and non_pg_tvars separation!")
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        grads = tf.gradients(self._cost, pg_tvars + non_pg_tvars)
        # Globally clip_gradients treatment.
        # grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
        grads_pg = [grads[i] for i in range(len(pg_tvars))]
        _start = len(pg_tvars)
        grads_non_pg = [grads[i + _start] for i in range(len(non_pg_tvars))]
        tf.summary.scalar("pg_grad_norm", tf.global_norm(grads_pg))
        tf.summary.scalar("nonpg_grad_norm", tf.global_norm(grads_non_pg))
        # Only clip on the grads_non_pg, instead of global treatment.
        grads_non_pg, _ = tf.clip_by_global_norm(grads_non_pg,
                                                 config.max_grad_norm)
        optimizer_pg = tf.contrib.opt.LazyAdamOptimizer(self._lr / 100.)
        optimizer_nonpg = tf.train.GradientDescentOptimizer(self._lr)
        train_op_pg = optimizer_pg.apply_gradients(zip(grads_pg, pg_tvars))
        train_op_nonpg = optimizer_nonpg.apply_gradients(
            zip(grads_non_pg, non_pg_tvars),
            global_step=tf.train.get_or_create_global_step())
      self._train_op = tf.group(train_op_pg, train_op_nonpg)
    elif config.optimizer == "scheduled_sgd":
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        grads = tf.gradients(self._cost, non_pg_tvars)
        tf.summary.scalar("global_grad_norm", tf.global_norm(grads))
        grads, _ = tf.clip_by_global_norm(grads,
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, non_pg_tvars),
            global_step=tf.train.get_or_create_global_step())
    else:
      self._train_op = tf.contrib.layers.optimize_loss(
          loss=self._cost,
          global_step=tf.train.get_or_create_global_step(),
          learning_rate=tf.convert_to_tensor(self._lr),
          optimizer=util.get_optimizer(config.optimizer),
          variables=non_pg_tvars,
          clip_gradients=float(config.max_grad_norm),
          summaries=["learning_rate", "loss", "global_gradient_norm"])
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells without Wrapper."""
    rnn_bn = False  # Seems not helpful..
    res_dropout_scheme = "x_fx"  # x_fx: dropout combined, fx: dropout only fx.
    res_div = 2.
    if not config.rnn_residual:
      res_dropout_scheme = "none"
    init_sates = []
    final_states = []
    with tf.variable_scope("RNN", reuse=not is_training):
      for l in range(config.num_layers):
        with tf.variable_scope("layer_%d" % l):
          cell = self._get_lstm_cell(config, is_training)
          initial_state = cell.zero_state(self.batch_size_final, data_type())
          init_sates.append(initial_state)
          state = init_sates[-1]
          if res_dropout_scheme == "fx" and l > 0:
            inputs, residual = inputs
          inputs_raw = inputs
          if rnn_bn and l > 0:
            inputs = tf.layers.batch_normalization(inputs, training=is_training)
          if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
          if res_dropout_scheme == "fx" and l > 0:
            inputs = (inputs + residual) / res_div
          inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
          outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                                      initial_state=init_sates[-1])
          final_states.append(state)
          outputs = [tf.expand_dims(output, 1) for output in outputs]
          outputs = tf.concat(outputs, 1)
          if res_dropout_scheme == "x_fx":
            outputs = (outputs + inputs_raw) / res_div
          if res_dropout_scheme == "fx":
            inputs = (outputs, inputs_raw)
          else:
            inputs = outputs
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    self._initial_state = tuple(init_sates)
    state = tuple(final_states)
    if rnn_bn:
      outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if is_training and config.keep_prob < 1:
      outputs = tf.nn.dropout(outputs, config.keep_prob)
    if res_dropout_scheme == "fx":
      outputs = (
          outputs + tf.reshape(inputs[1], [-1, config.hidden_size])) / res_div
    return outputs, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def nll(self):
    return self._nll

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  if FLAGS.optimizer in ["sgd", "scheduled_sgd", "momentum", "mixed"]:
    optimizer = FLAGS.optimizer
    learning_rate = FLAGS.learning_rate if FLAGS.learning_rate is not None else 1.0
  elif FLAGS.optimizer == "lazy_adam":
    optimizer = "lazy_adam"
    learning_rate = FLAGS.learning_rate if FLAGS.learning_rate is not None else 1e-3
  elif FLAGS.optimizer == "full_adam":
    optimizer = "adam"
    learning_rate = FLAGS.learning_rate if FLAGS.learning_rate is not None else 1e-3
  else:
    raise ValueError("FLAGS.optimizer = {} is unknown.".format(FLAGS.optimizer))
  test_summary = FLAGS.test_summary
  max_grad_norm = 5 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
  weight_decay = 0. if FLAGS.weight_decay is None else FLAGS.weight_decay
  emb_lowrank_dim = 0. if FLAGS.emb_lowrank_dim is None else FLAGS.emb_lowrank_dim
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 100 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  global vocab_size
  rnn_mode = BLOCK
  rnn_residual = False if FLAGS.rnn_residual is None else FLAGS.rnn_residual
  shared_embedding = False if FLAGS.shared_embedding is None else FLAGS.shared_embedding

  # D = 2
  # K = int(np.ceil(vocab_size**(1/float(D))))
  # code_type = "compact"
  D = 10 if FLAGS.D is None else FLAGS.D
  K = 100 if FLAGS.K is None else FLAGS.K
  code_type = "redundant" if FLAGS.code_type is None else FLAGS.code_type

  # Encoder params.
  disable_encoding = False
  # ec_code_generator = "preassign"
  # ec_code_generator = "gumbel_softmax"
  # ec_code_generator = "STE_threshold"
  ec_code_generator = "STE_argmax" if FLAGS.ec_code_generator is None else FLAGS.ec_code_generator
  ec_logits_bn = 0. if FLAGS.ec_logits_bn is None else FLAGS.ec_logits_bn
  ec_emb_baseline = False if FLAGS.ec_emb_baseline is None else FLAGS.ec_emb_baseline
  ec_emb_baseline_reg = 0. if FLAGS.ec_emb_baseline_reg is None else FLAGS.ec_emb_baseline_reg
  ec_emb_baseline_reg2 = 0. if FLAGS.ec_emb_baseline_reg2 is None else FLAGS.ec_emb_baseline_reg2
  ec_emb_baseline_reg3 = 0. if FLAGS.ec_emb_baseline_reg3 is None else FLAGS.ec_emb_baseline_reg3
  ec_emb_baseline_dropout = 0. if FLAGS.ec_emb_baseline_dropout is None else FLAGS.ec_emb_baseline_dropout
  ec_STE_softmax_transform = True  if FLAGS.ec_STE_softmax_transform is None else FLAGS.ec_STE_softmax_transform  # Only applicable for STE ec_code_generator.
  ec_entropy_reg = 0. if FLAGS.ec_entropy_reg is None else FLAGS.ec_entropy_reg
  ec_temperature_decay_method = "none" if FLAGS.ec_temperature_decay_method is None else FLAGS.ec_temperature_decay_method
  ec_temperature_decay_steps = 1000 if FLAGS.ec_temperature_decay_steps is None else FLAGS.ec_temperature_decay_steps
  ec_temperature_decay_rate = 1. if FLAGS.ec_temperature_decay_rate is None else FLAGS.ec_temperature_decay_rate
  ec_temperature_init = 1. if FLAGS.ec_temperature_init is None else FLAGS.ec_temperature_init
  ec_temperature_lower_bound = 1e-5 if FLAGS.ec_temperature_lower_bound is None else FLAGS.ec_temperature_lower_bound
  ec_hard_code_output = True if FLAGS.ec_hard_code_output is None else FLAGS.ec_hard_code_output  # Only applicable for not preassign.
  ec_code_emb_dim = 200 if FLAGS.ec_code_emb_dim is None else FLAGS.ec_code_emb_dim
  ec_shared_coding = True if FLAGS.ec_shared_coding is None else FLAGS.ec_shared_coding
  ec_code_dropout = 0. if FLAGS.ec_code_dropout is None else FLAGS.ec_code_dropout
  ec_aggregator = "rnn" if FLAGS.ec_aggregator is None else FLAGS.ec_aggregator
  ec_cnn_num_layers = 1 if FLAGS.ec_cnn_num_layers is None else FLAGS.ec_cnn_num_layers
  ec_cnn_residual = True if FLAGS.ec_cnn_residual is None else FLAGS.ec_cnn_residual
  ec_cnn_pooling = "max" if FLAGS.ec_cnn_pooling is None else FLAGS.ec_cnn_pooling
  ec_fnn_num_layers = 1 if FLAGS.ec_fnn_num_layers is None else FLAGS.ec_fnn_num_layers
  ec_fnn_hidden_size = 500 if FLAGS.ec_fnn_hidden_size is None else FLAGS.ec_fnn_hidden_size
  ec_fnn_hidden_actv = "relu" if FLAGS.ec_fnn_hidden_actv is None else FLAGS.ec_fnn_hidden_actv
  ec_cnn_filters = 500 if FLAGS.ec_cnn_filters is None else FLAGS.ec_cnn_filters
  ec_cnn_kernel_size = 3 if FLAGS.ec_cnn_kernel_size is None else FLAGS.ec_cnn_kernel_size
  ec_cnn_hidden_actv = "relu" if FLAGS.ec_cnn_hidden_actv is None else FLAGS.ec_cnn_hidden_actv
  ec_rnn_num_layers = 1 if FLAGS.ec_rnn_num_layers is None else FLAGS.ec_rnn_num_layers
  ec_rnn_hidden_size = 500 if FLAGS.ec_rnn_hidden_size is None else FLAGS.ec_rnn_hidden_size
  ec_rnn_additive_pooling = True if FLAGS.ec_rnn_additive_pooling is None else FLAGS.ec_rnn_additive_pooling
  ec_rnn_residual = False if FLAGS.ec_rnn_residual is None else FLAGS.ec_rnn_residual
  ec_rnn_dropout = 0. if FLAGS.ec_rnn_dropout is None else FLAGS.ec_rnn_dropout
  ec_rnn_trainable_init_state = True if FLAGS.ec_rnn_trainable_init_state is None else FLAGS.ec_rnn_trainable_init_state
  ec_rnn_bidirection = False if FLAGS.ec_rnn_bidirection is None else FLAGS.ec_rnn_bidirection
  ec_emb_autoencoding = False if FLAGS.ec_emb_autoencoding is None else FLAGS.ec_emb_autoencoding
  ec_emb_transpose_layers = 1 if FLAGS.ec_emb_transpose_layers is None else FLAGS.ec_emb_transpose_layers
  ec_emb_transpose_dim = 100 if FLAGS.ec_emb_transpose_dim is None else FLAGS.ec_emb_transpose_dim
  ec_emb_transpose_actv = "tanh" if FLAGS.ec_emb_transpose_actv is None else FLAGS.ec_emb_transpose_actv


class MediumConfig(SmallConfig):
  """Medium config."""
  init_scale = 0.05
  max_grad_norm = 5 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 12 if FLAGS.use_recoding else 6
  max_max_epoch = 39 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
  keep_prob = 0.5
  lr_decay = 0.6 if FLAGS.use_recoding else 0.8
  batch_size = 20
  global vocab_size
  rnn_mode = BLOCK


class LargeConfig(SmallConfig):
  """Large config."""
  init_scale = 0.04
  max_grad_norm = 10 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
  keep_prob = 0.35
  lr_decay = 0.6 if FLAGS.use_recoding else 1 / 1.15
  batch_size = 20
  global vocab_size
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  global vocab_size
  rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False, mode=TRAIN_MODE):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  nlls = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "nll": model.nll,
      "final_state": model.final_state
  }

  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    nll = vals["nll"]
    state = vals["final_state"]

    costs += cost
    nlls += nll
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f cost %.3f perplexity: %.3f speed: %.0f wps" %
            (step*1./model.input.epoch_size, costs / iters, np.exp(nlls/iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  # Examine at the end of the epoch.
  if hasattr(model, "query_codes") and mode == TRAIN_MODE and (
      model.query_codes is not None):
    fetches_ = {"query_codes": model.query_codes}
    fetches_["query_input_emb"] = model.query_input_emb
    fetches_["query_output_emb"] = model.query_output_emb
    vals = session.run(fetches_, feed_dict)
    query_codes = vals["query_codes"]
    if len(query_codes.shape) > 2:
      query_codes = np.argmax(query_codes, axis=-1)  # (vocab_size, D)
    if FLAGS.code_save_filename:
      if FLAGS.save_path is None:
        path = './{}_{}.pkl'.format(
            FLAGS.code_save_filename, int(time.time()))
      else:
        path = '{}/{}_{}.pkl'.format(
            FLAGS.save_path, FLAGS.code_save_filename, int(time.time()))
      with open(path, 'w') as fp:
        pickle.dump(query_codes, fp, 2)

      # Save input_emb and output_emb. Need to disable query_codes is not None.
      if False:
        with open(path.replace("code", "input_emb"), 'w') as fp:
          pickle.dump(
              {"vocab": id2word, "embs": vals["query_input_emb"]}, fp, 2)
        with open(path.replace("code", "output_emb"), 'w') as fp:
          pickle.dump(
              {"vocab": id2word, "embs": vals["query_output_emb"]}, fp, 2)

    # Compute collision rate.
    query_codes_set = set([tuple(each) for each in query_codes])
    print("Ratio of unique elements in code space: {}".format(
        float(len(query_codes_set)) / query_codes.shape[0]))

    # Print collision words.
    cs = ["-".join(each) for each in query_codes.astype('str').tolist()]
    cs_cnt = Counter(cs)
    duplicated = defaultdict(list)
    for i, code_ in enumerate(cs):
      if cs_cnt[code_] >= 1:  # use >=1 to save all codes, use >1 to save only duplicated ones.
        duplicated[code_].append(id2word[i])
    if FLAGS.save_path is None:
      path = './'
    else:
      path = FLAGS.save_path
    if False:  # Print word collisions.
      with open("{}/duplicated_{}".format(path, time.asctime()), "w") as fp:
        for code_ in sorted(duplicated):
          fp.write("{}\t\t{}\n".format(code_, duplicated[code_]))

    # Find out how many codes have been changed since last time here.
    global last_cs
    if last_cs is not None:
      bit_diff = lambda x, y: np.sum(
          [0 if s == t else 1 for s, t in zip(x.split('-'), y.split('-'))])
      num_code_changed = np.sum(
          [0 if curt_c == past_c else 1 for curt_c, past_c in zip(cs, last_cs)])
      num_bit_changed = np.sum(
          [bit_diff(curt_c, past_c) for curt_c, past_c in zip(cs, last_cs)])
      print("Percent of code changed {}".format(
          num_code_changed / float(len(cs))))
      print("Percent of bit changed per changed code {}".format(
          0 if num_code_changed == 0 else (
              num_bit_changed / float(num_code_changed) /  model._config.D)))
    last_cs = cs

  return np.exp(nlls / iters)


def get_config(verbose=False):
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if verbose:
    config_attrs = [a for a in inspect.getmembers(config) if not (
        a[0].startswith('__') and a[0].endswith('__'))]
    print(config_attrs)
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.dataset, FLAGS.data_path, True)
  train_data, valid_data, test_data, _vocab_size, _id2word = raw_data
  global id2word, vocab_size
  id2word = _id2word
  vocab_size = _vocab_size

  config = get_config(verbose=True)
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  # Load pretrained embedding.
  if FLAGS.emb_load_filename is not None and FLAGS.emb_load_filename != "":
    with open(FLAGS.emb_load_filename) as fp:
      pretrained_emb = pickle.load(fp)
  else:
    pretrained_emb = None

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True,
                     config=config,
                     input_=train_input,
                     vocab_size=vocab_size,
                     pretrained_emb=pretrained_emb)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False,
                          config=config,
                          input_=valid_input,
                          vocab_size=vocab_size,
                          pretrained_emb=pretrained_emb)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False,
                         config=eval_config,
                         input_=test_input,
                         vocab_size=vocab_size,
                         pretrained_emb=pretrained_emb)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}

    print_at_beginning(config)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path,
                             save_model_secs=FLAGS.save_model_secs,
                             save_summaries_secs=10)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    with sv.managed_session(config=config_proto) as session:
      if not FLAGS.use_recoding and pretrained_emb is not None:
        print("Use pretrained_emb without KD encoding.")
        session.run(m.using_pretrained_embs_on_run_op)
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True, mode=TRAIN_MODE)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, mode=VALID_MODE)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest, mode=TEST_MODE)
      print("Test Perplexity: %.3f" % test_perplexity)

      # Save word embedding.
      if FLAGS.emb_save_filename is not None and FLAGS.emb_save_filename != "":
        embs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope="Model/embedding")
        if len(embs) != 1:
          raise ValueError("embs are {}".format(embs))
        embs = session.run(embs)
        with open(FLAGS.emb_save_filename, "wb") as fp:
          pickle.dump(embs[0], fp, 2)

      if FLAGS.save_path and sv.saver is not None:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session,
                      os.path.join(FLAGS.save_path, "model"),
                      global_step=sv.global_step)


def reload_embedding_after_checkpoint_recover(pretrained_emb):
  embs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope="Model/embedding")
  assert len(embs) == 1, len(embs)
  return tf.assign(embs[0], pretrained_emb)

if __name__ == "__main__":
  tf.app.run()
