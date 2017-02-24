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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import behaviour_clone

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './data/data_cb/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', './data/data_cb/',
                           """validation data directory""")
tf.app.flags.DEFINE_string('checkpoint_dir', './data/data_cb/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs',  60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 200,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, valid_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # true_count = 0  # Counts the number of correct predictions.
      # accuracy_sum = 0
      # total_sample_count = num_iter * FLAGS.batch_size
      sum_rmse = 0
      step = 0
      while step < num_iter and not coord.should_stop():
        sum_rmse += sess.run(valid_op)
        # true_count += np.sum(predictions)
        step += 1
      rmse = float(sum_rmse/num_iter)
      # Compute precision @ 1.
      # accuracy = accuracy_sum / total_sample_count
      print('%s: rmse = %.3f' % (datetime.now(), rmse))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='rmse', simple_value=rmse)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  # with tf.Graph().as_default(),tf.device('/cpu:0') as g:
  with tf.device('/cpu:0') as g:

    global_step = tf.Variable(0, trainable=False)
    # # Get images and labels for CIFAR-10.
    # eval_data = FLAGS.eval_data == 'test'
    # images, labels = cifar10.inputs(eval_data=eval_data)
    
    observations_batch_eval, actions_batch_eval = behaviour_clone.inputs(eval_data=True)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    # logits = cifar10.inference(images)
    pred_actions = behaviour_clone.inference(observations_batch_eval)

    # Calculate predictions.
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)
    eval_op = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(actions_batch_eval, pred_actions))))
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        behaviour_clone.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, valid_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

  evaluate()


if __name__ == '__main__':
  tf.app.run()
