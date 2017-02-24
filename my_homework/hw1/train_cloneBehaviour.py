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

BATCH_SIZE = 128
EPO_STEP = 50000
TEST_STEP = 200
num_rollouts = 20
# ===========================
#   Utility Parameters
# ===========================
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/'
DAGGAR = True

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
        net = tflearn.fully_connected(inputs, 128)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.fully_connected(inputs, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
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
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
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
    reward_predict = tf.Variable(0.)
    tf.summary.scalar('Predict reward', reward_predict)

    summary_vars = [loss,rmse,reward_predict]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================
def train(sess, env, CloneBehavior, expert_data):

		# summary_ops, summary_vars = build_summaries()
		sess.run(tf.global_variables_initializer())

		# load model if have
		# saver = tf.train.Saver()
		# checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)

		# if checkpoint and checkpoint.model_checkpoint_path:
		# 	saver.restore(sess, checkpoint.model_checkpoint_path)
		# 	print ("Successfully loaded:", checkpoint.model_checkpoint_path)

		# else:
		# 	print ("Could not find old network weights")
		
		# writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

		i=0
		policy_fn = load_policy.load_policy("./experts/Humanoid-v1.pkl")
		max_steps = env.spec.timestep_limit
		while i < EPO_STEP:
			i+=1
			data_size = expert_data['observations'].shape[0]
			batch_idx = np.random.randint(data_size, size = BATCH_SIZE)
			observations_batch = expert_data['observations'][batch_idx,:]
			actions_batch = expert_data['actions'][batch_idx,:]
		# mix randomly??

			if DAGGAR and i%200==0:
				obs_dagger_batch = []
				act_dagger_batch = []	
				print "Daggering"
				for j in range(num_rollouts):
					obs = env.reset()
					done = False
					steps = 0
					while not done:
						action = CloneBehavior.predict(prepro(obs))
						action_expert = policy_fn(obs[None,:])
						obs_dagger_batch.append(obs)
						act_dagger_batch.append(action_expert[0])
						obs, _, done ,_ = env.step(action)
						steps += 1
						if steps >= max_steps:
							break
				
				obs_dagger_batch = np.array(obs_dagger_batch)
				act_dagger_batch = np.array(act_dagger_batch)
				# print obs_dagger_batch.shape
				expert_data['observations'] = np.concatenate((expert_data['observations'], obs_dagger_batch), axis=0)
				expert_data['actions'] = np.concatenate((expert_data['actions'], act_dagger_batch), axis=0)
				# print ("After daggering the size of data become", expert_data['actions'].shape)
				# training the batch:
			_, _, loss = CloneBehavior.train(observations_batch, actions_batch)

			if i%100 ==0:
				print('epoch', i, 'loss', loss)

		# print('loading and building expert policy')
		# policy_fn = load_policy.load_policy("./experts/Humanoid-v1.pkl")
		# print('loaded and built')
		# max_steps = env.spec.timestep_limit
		# observation = env.reset()
		# done = False
		# batch_num = 0
		# reward_total = 0
		# r_pre = 0
		# for batch_num in range(EPO_STEP):
		# 	i += 1
		# 	if i < (EPO_STEP - TEST_STEP):
		# 		if i > 20 and (i%10) == 1:
		# 			#train batches
		# 			# batch_num+=1
		# 			time_gap = time.time() - tic
		# 			observations_batch = np.asarray(observations_batch)
		# 			actions_batch = np.asarray(actions_batch)
		# 			CloneBehavior.train(observations_batch,actions_batch)
		# 			# evaluatio	n
		# 			observation_once = prepro(observation)
		# 			action_once = policy_fn(observation[None,:])
		# 			action_once = prepro(action_once[0])

		# 			loss = CloneBehavior.rmse(observation_once,action_once)
		# 			action_predict = CloneBehavior.predict(observation_once)
		# 			# do the evaluation:
		# 			_, reward_predict ,_ ,_ = env.step(action_predict)
		# 			reward_total += reward_predict
		# 			print('| Reward: %.2f' %(reward_total) , '| Time: %.2f' %(time_gap),'| Loss: %.2f' %(loss))

		# 			summary_str = sess.run(summary_ops, feed_dict={
		# 				summary_vars[1]: loss,
		# 				summary_vars[2]: reward_total,
		# 				})
		# 			writer.add_summary(summary_str, i)
		# 			writer.flush()
					
		# 			observations_batch = []
		# 			actions_batch = []
		# 		else:
		# 			action = policy_fn(observation[None,:])
		# 			action = action[0]
		# 			observations_batch.append(observation) # obtain one time observations
		# 			actions_batch.append(action)
		# 			observation, r, done, _ = env.step(action)
		# 			if i%100 ==0: print("%i/%i"%(i, max_steps))
		# 	else:
		# 		print("Predicting....")
		# 		observation_once = prepro(observation)
		# 		action_predict = CloneBehavior.predict(observation_once)
		# 		observation, reward_predict, _, _ = env.step(action_predict)
		# 		env.render()
		# 		r_pre += reward_predict
		# 		print('| Reward: %.2f' %(r_pre))
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
def test(sess, cloneBehavior, env):
	returns = []
	for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            max_steps = env.spec.timestep_limit
            while not done:
                action = cloneBehavior.predict(prepro(obs))
                # observations.append(obs)
                # actions.append(action[0])
                action = [action]
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            print('iter',i,'total reward',totalr)
            returns.append(totalr)

	print('returns', returns)
	print('mean return', np.mean(returns))
	print('std of return', np.std(returns))

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # =============================
    #	load expert data
    # =============================
    expert_data={}
    expert_data['actions'] = np.load("./expert_actions.npy")
    expert_data['observations'] = np.load("./expert_obs.npy")
    print ("loaded data....")
    print("observations dim:",expert_data['observations'].shape)
    print("actions dim:",expert_data['actions'].shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        env = gym.make('Humanoid-v1')
        
        state_dim = env.observation_space.shape[0] #hot vector:376
        action_dim = env.action_space.shape[0] #hot vector:17
        

        cloneBehavior = CloneBehavior(sess, state_dim, action_dim, LEARNING_RATE)

        train(sess, env, cloneBehavior, expert_data)
        test(sess, cloneBehavior, env)

if __name__ == '__main__':
    tf.app.run()
