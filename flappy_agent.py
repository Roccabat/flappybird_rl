import argparse, sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym
from gym.wrappers import Monitor
import flappy_env

def define_vector_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(LeakyReLU(0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(0.2))
    model.add(Dense(8))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nb_actions))
    return model

def define_image_model(input_shape, nb_actions):        
    model = Sequential()
    model.add(Conv2D(8, (8, 8), strides=(2, 2), input_shape=input_shape, data_format="channels_first"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(16, (4, 4), strides=(2, 2), data_format="channels_first"))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU(0.2))
    model.add(Dense(8))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nb_actions))
    return model




def main():
    ENV_NAME = 'FlappyB-v0'

    # Get the environment
    env = gym.make(ENV_NAME, env_state=FLAGS.env_state)

    # Create the monitors that record the testing and training videos
    recorded_training = [10, 100, 1000]
    env_train = Monitor(env, "./training_video", force=True, video_callable= lambda episode_id: episode_id in recorded_training)
    env_test = Monitor(env, "./testing_video", force=True, video_callable= lambda episode_id: True)

    # Get the number of actions from the environement
    nb_actions = env.action_space.n

    # Choose the memory_len and the model we will train
    # As we can train on a vector representing the world or on the screen image of the game, we have two different architectures
    if FLAGS.env_state == "vector":
        # Choose the memory length, here one vector is enough to represent entirely the world so we choose a memory length of one
        memory_windows_len = 1

        model = define_vector_model((memory_windows_len,) + env.observation_space.shape, nb_actions)
        print(model.summary())
        learning_rate = 1e-3
    elif FLAGS.env_state == "image":
        # Choose the memory length, here more than one image is needed to get the speed of the bird
        memory_windows_len = 3

        model = define_image_model((memory_windows_len,) + env.observation_space.shape, nb_actions)
        print(model.summary())
        learning_rate = 1e-3
    else:
        print("Warning the env state is not recognised, please select image or vector")
        return




    # We define the memory for experience replay and the policy for our flappy bird agent
    memory = SequentialMemory(limit=20000, window_length=memory_windows_len)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.05, value_min=.001, value_test=.00, nb_steps=25000)

    # We compile our agent
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200, target_model_update=0.1, policy=policy, train_interval=1)
    agent.compile(Adam(lr=learning_rate), metrics=['mae'])

    # If required, we restore the weights
    if FLAGS.restore == "y":
        agent.load_weights('agent_{}_weights.h5f'.format(ENV_NAME))

    # Now it's time to learn something!
    # We can always safely abort the training prematurely using Ctrl + C
    try:
        agent.fit(env_train, nb_steps=25000, visualize=False, verbose=2)
    except:
        pass

    # After training is done, we save the final weights
    agent.save_weights('agent_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes
    agent.test(env_test, nb_episodes=10, visualize=False)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_state', type=str, default='vector', help='Choose the kind of state returned by the environement: "vector" or "image"')
  parser.add_argument('--restore', type=str, default='n', help='Restore old weights? (y/n)')
  FLAGS, unparsed = parser.parse_known_args()
  main()
