# Flappy Bird with Reinforcement Learning

## What is flappybird_rl?

This repository provides the python code for the training of a basic Reinforcement Learning agent to play the Flappy Bird game.
More details about this project can be found in the related Medium blog post: [comming soon].

![](maths.gif)

A Reinforcement Learning project requires to main pieces: an environement and an agent. In our case the environement is the Flappy Bird game and we will train an agent to play this game. Thus, this repository contains:
- code and explanations to wrap a given environement (here from PyGame Learning Environment) to turn it into an OpenAI Gym environement and to make it easily compatible with a keras-rl agent
- code and explanations to train an agent with respect to a given Gym Environment (using the keras-rl library)
- a dockerfile to build a docker image that will contains all the required dependancies to run the code of this repository



## About the environement

In order to set up the environment, you will need [PyGame Learning Environment (PLE)](https://pygame-learning-environment.readthedocs.io), [Gym](https://gym.openai.com) and CCC to be installed.

The Flappy Bird environement can be found in the PyGame Learning Environment (PLE) library. In order to be able to use the keras-rl library to define your agent easily, you can first wrap this environement so that to obtain a standard OpenAi Gym environment (see code of env.py for details). The wrapper should inherit from the Gym.Env class and must implement at least N functions: XXX that makes ZZZ, ...

For the state, we have defined our wrapper so that to be able to train the agent on two variants of the environement:
- in the first case (screen_state=False) the state returned by the environement is a vector that contains (...)
- in the second case (screen_state=True) the state returned by the environement is directly a 2D array that corresponds to the "game screen" (resized and transformed in black and white to make the training time reasonable)

For the reward, we simply give a -200 reward to Flappy when it dies.

Once the environment wrapped, you need to...

Of course, this can be applied to other environements from PLE, from other libraries or that you developped by yourself.



## About the agent

In order to set up the agent, you will need tensorflow, [keras-rl](https://github.com/keras-rl/keras-rl) and CCC to be installed.

Once the environment has been turned into a Gym environment, you can use keras and keras-rl libraries easily to define and train your agent. In order to set up your agent for its training, you need to define...



## Some results and numbers

| | Vector based Flappy | Screen based Flappy |
|-|---------------------|---------------------|
| Number of parameters | 10 | 1000 |
| Training time | 10s | 100s |
| Best score | 123 | 45 |
