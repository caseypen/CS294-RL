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
import glob
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the  data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 800
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

ACTION_DIM = 17
OBSERVATION_DIM = 376

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'expert_training_data.tfrecords'

EVAL_FILE = 'expert_eval_valid.tfrecords'

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'observations': tf.FixedLenFeature([], tf.string),
          'actions': tf.FixedLenFeature([], tf.string)
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  observations = tf.decode_raw(features['observations'], tf.float64)
  actions = tf.decode_raw(features['actions'], tf.float64)

  print(actions)
  observations = tf.reshape(observations, [OBSERVATION_DIM])
  actions = tf.reshape(actions, [ACTION_DIM])
  
  
  observations = tf.cast(observations, tf.float32)
  actions = tf.cast(actions, tf.float32)
  # print(actions)
  # print(observations)

  # actions = tf.cast(features['actions'], tf.float32)

  return observations, actions


def _generate_image_and_label_batch(observations,actions, min_queue_examples, batch_size, shuffle):
  # Create a queue that shuffles the examples
  # min_queue_examples = 100
  num_preprocess_threads = 8
  if shuffle:
    observations_batch, actions_batch = tf.train.shuffle_batch(
        [observations, actions],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    observations_batch, actions_batch = tf.train.batch(
        [observations,actions],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('observations', observations_batch)
  # tf.image_summary('actions', actions_batch)

  return observations_batch, actions_batch 


def inputs(eval_data, data_dir, batch_size):
  """Construct input for  evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filename = os.path.join(data_dir, TRAIN_FILE)
    # filename2 = os.path.join(data_dir, TRAIN2_FILE)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filename = os.path.join(data_dir, EVAL_FILE)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  with tf.name_scope('input'):
    if not eval_data:
      filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    else:
      filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    observations, actions = read_and_decode(filename_queue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of left_images right_images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(observations, actions, 
      min_queue_examples, batch_size, shuffle=False)
