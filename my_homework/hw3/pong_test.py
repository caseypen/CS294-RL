import gym
benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]
env = gym.make(task.env_id)
# env = gym.make('Pong-ram-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        action = 2
        observation, reward, done, info = env.step(action)
        print action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break