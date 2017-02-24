#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import os
import glob

tf.app.flags.DEFINE_string('directory', './data/data_cb',
                           'Directory to expert clone data')
tf.app.flags.DEFINE_integer('validation_size', 100,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def convert_to(left_images, right_images, lidar_labels, name):
def convert_to(expert_data, name):    
  observations = expert_data['observations']
  print(observations.ndarray.dtype)
  actions = expert_data['actions']
  num_examples = actions.shape[0]
  if observations.shape[0] != num_examples:
    raise ValueError("Observatoin size %d does not match action size %d." %
                     (observations.shape[0], num_examples))
  else:
    print("number of training data is", num_examples)
  observations_dim = observations.shape[1]
  actions_dim = actions.shape[1]
  
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    observations_raw = observations[index].tostring()
    actions_raw = actions[index].tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'observations': _bytes_feature(observations_raw),
        'actions': _bytes_feature(actions_raw)
        }))
    writer.write(example.SerializeToString())

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        #seperate data:
        num_examples = len(observations)
        observations = np.array(observations)
        actions = np.array(actions)
        observations_train = observations[:num_examples*4/5]
        actions_train = actions[:num_examples*4/5]
        # print ("the number of training is",len(observations_train))
        observations_eval = observations[len(observations_train):]
        actions_eval = observations[len(observations_train):]
        expert_data_train = {'observations': observations_train,
                             'actions': actions_train}
        expert_data_eval = {'observations': observations_eval,
                            'actions': actions_eval}
        name_train = "expert_training_data"
        name_eval = "expert_eval_data"
        convert_to(expert_data_train, name_train)
        convert_to(expert_data_eval,name_eval)

if __name__ == '__main__':
    main()
