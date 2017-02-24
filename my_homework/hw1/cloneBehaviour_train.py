import tensorflow as tf
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import time
import load_policy
import tf_util
import gym

# ==========================
#   Training Parameters
# ==========================

# Max episode length    
MAX_EP_STEPS = 100
# Base learning rate for the Actor network
#ACTOR_LEARNING_RATE = 1e-4
# Base learning rate for the Critic Network
LEARNING_RATE = 1e-4

BATCH_SIZE = 10
EPO_STEP = 25000
TEST_STEP = 5000
# ===========================
#   Utility Parameters
# ===========================
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/'


class CloneBehavior(object):
    """ 
    Input to the network is the state, output is Q(s).
    The action is updated by epsilon-gradient

    """
    def __init__(self, sess, observations_dim, actions_dim, learning_rate):
			self.sess = sess
			self.s_dim = observations_dim 
			self.a_dim = actions_dim 
			self.learning_rate = learning_rate

			# Create the critic network
			self.inputs, self.out = self.create_cb_network()

			self.network_params = tf.trainable_variables()

			# action taken
			self.actions = tf.placeholder(tf.float32, [None, self.a_dim])
			# self.obs = tf.placeholder(tf.float64,self.s_dim[0])

			# Define loss and optimization Op
			# self.dqn_global_step = tf.Variable(0, name='dqn_global_step', trainable=False)

			self.loss = tflearn.mean_square(self.out, self.actions)
			# self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.dqn_global_step)
			self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def create_cb_network(self):
        inputs = tflearn.input_data(shape=[None,self.s_dim]) # placeholder
        net = tflearn.fully_connected(inputs, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        # # action = tflearn.input_data(shape=[None,self.a_dim]) # not take in action when calculating Q 
        # net = tflearn.conv_2d(inputs, 8, 3, activation='relu', name='conv1') # inputs must be 4D tensor
        # # net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv2')
        # net = tflearn.fully_connected(inputs, 64, activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)

        # # Add the action tensor in the 2nd hidden layer
        # # Use two temp layers to get the corresponding weights and biases
        # # t1 = tflearn.fully_connected(net, 64)
        # # t2 = tflearn.fully_connected(action, 64)

        # # net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tflearn.fully_connected(net, 32, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim)
        return inputs, out

    def train(self, observations, actions):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: observations,
            self.actions: actions,
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
    def rmse(self, obs, action):
    	return self.sess.run(self.loss, feed_dict={
    		self.inputs: obs,
    		self.actions: action
    	})

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    # training
    loss = tf.Variable(0.)
    tf.summary.scalar('loss', loss)
    # predicting
    rmse = tf.Variable(0.)
    tf.summary.scalar('rmse', rmse)
    reward_predict = tf.Variable(0.)
    tf.summary.scalar('Predict reward', reward_predict)
    expert_reward = tf.Variable(0.)
    tf.summary.scalar('Expert reward', expert_reward)

    summary_vars = [loss,rmse,reward_predict,expert_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================
def train(sess, env, CloneBehavior, global_step):

		summary_ops, summary_vars = build_summaries()
		sess.run(tf.global_variables_initializer())

		# load model if have
		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)

		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print ("Successfully loaded:", checkpoint.model_checkpoint_path)
			print("global step: ", global_step.eval())

		else:
			print ("Could not find old network weights")
		
		writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

		i = global_step.eval()
		i=0

		tic = time.time()

		observations_batch = []
		actions_batch = []
		# env = gym.make('Humanoid-v1')
		print('loading and building expert policy')
		policy_fn = load_policy.load_policy("./experts/Humanoid-v1.pkl")
		print('loaded and built')
		max_steps = env.spec.timestep_limit
		observation = env.reset()
		done = False
		batch_num = 0
		reward_total = 0 # agent reward
		r_total = 0      # expert reward
		r_pre = 0        # predict
		for batch_num in range(EPO_STEP):
			i += 1
			if i < (EPO_STEP - TEST_STEP):
				#train batches
				# batch_num+=1
				time_gap = time.time() - tic
				observation_batch = prepro(observation)
				action = policy_fn(observation[None,:])
				action_batch = prepro(action[0])
				CloneBehavior.train(observation_batch,action_batch)
				observation, r, done, _ = env.step(action)
				if i%20 == 0:	
					loss = CloneBehavior.rmse(observation_batch,action_batch)
					action_predict = CloneBehavior.predict(observation_batch)
					# do the evaluation:
					_, reward_predict ,_ ,_ = env.step(action_predict)
					reward_total += reward_predict
					r_total += r
					print('| Reward: %.2f' %(reward_total) , '| Time: %.2f' %(time_gap),'| Loss: %.2f' %(loss),'| Expert R: %.2f' %(r_total))

					summary_str = sess.run(summary_ops, feed_dict={
						summary_vars[0]: loss,
						summary_vars[2]: reward_total,
						summary_vars[3]: r_total
						})
					writer.add_summary(summary_str, i)
					writer.flush()
				if i%100 ==0: print("%i/%i"%(i, max_steps))
			else:
				print("Predicting....")
				observation_once = prepro(observation)
				action_predict = CloneBehavior.predict(observation_once)
				observation, reward_predict, _, _ = env.step(action_predict)
				env.render()
				r_pre += reward_predict
				print('| Reward: %.2f' %(r_pre))
				# if i >= max_steps:
				# 	break
				# print('Expert reward %.2f' %(r))
			# if i > TEST_STEP:
			# 	action_predict = CloneBehavior.predict(observation)
			# 	observation, reward_predict, _, _ = env.step(action_predict)
			# 	reward_total += reward_predict
			# 	print ('| Reward: %.2f' %(reward_total) '| Time: %.2f' %(time_gap))
def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = np.reshape(state,(1,state.shape[0]))
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return state
def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # env = sim_env(MAP_SIZE, PROBABILITY) # create
        env = gym.make('Humanoid-v1')
        # s = env.reset()
        # print(s)
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)

        # state_dim = env.nS
        state_dim = env.observation_space.shape[0] #hot vector:376
        print('state_dim:',state_dim)
        action_dim = env.action_space.shape[0] #hot vector:17
        print('action_dim:',action_dim)

        cloneBehavior = CloneBehavior(sess, state_dim, action_dim, LEARNING_RATE)

        train(sess, env, cloneBehavior, global_step)

if __name__ == '__main__':
    tf.app.run()
