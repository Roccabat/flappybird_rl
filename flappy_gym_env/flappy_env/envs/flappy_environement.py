import gym
from gym import spaces
from gym.utils import seeding

from ple import PLE
from ple.games.flappybird import FlappyBird

import numpy as np
import cv2

VECTOR_STATE = 0
IMAGE_STATE = 1

class FlappyEnv(gym.Env):
    """
    Description:
        This environement wraps for gym the implementation of the flappy bird game as it is working on the PyGame Learning Environment
    Observation space for "vector" state: 
        Type: Box(8)
        Num	Observation                                     Min         Max
        0	player y position                               -10         500
        1	player velocity                                 -10         10
        2	distance to the next pipe                       0           400
        3	height of the top of the next pipe              0           500
        4	height of the bottom of the next pipe           0           500
        5	distance to the next next pipe                  0           800
        6	height of the top of the next next pipe         0           500
        7	height of the bottom of the next next pipe      0           500
    
    Observation space for "image" state: 
        Type: Box(image shape)
        In this case the environement returns a np array that represent the current screen image that has been resized and converted into black and white
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	flap
        1	wait
    
    Reward:
        Reward is 0 for every step taken, when dying the reward is -250 and when passing a pipe the reward is 150
    
    Episode Termination:
        The bird touches anything
    
    Note: 
        Flap velocity is not cumulated, flapping X times does not make the player goes up X times faster
        The y axis is inverted meaning that a negative velocity makes the player goes up
    """

    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self, env_state="vector"):
        if env_state == "vector":
            self.env_state = VECTOR_STATE
        elif env_state == "image":
            self.env_state = IMAGE_STATE
        else:
            self.env_state = VECTOR_STATE
            print("Warning: the environement state type is unrecognized. The default 'vector' state will be used.")
        
        # Define what game is going to be used and create an instance
        self.game = FlappyBird()
        self.game_instance = PLE(self.game, fps=30, display_screen=True)

        # The two possible actions are flap and stay still
        self.action_space = spaces.Discrete(2)

        # The observation space depends on the way we want to train our agent
        if self.env_state == IMAGE_STATE:
            # To get the image shape we take the output of the _get_obs() function 
            image = self._get_obs()
            self.observation_space = spaces.Box(low=0, high=255, shape=image.shape, dtype=np.uint8)
        if self.env_state == VECTOR_STATE:
            self.observation_space = spaces.Box(low=np.array([-10, -10, 0, 0, 0, 0, 0, 0]), high=np.array([500, 10, 400, 500, 500, 800, 500, 500]))
        
        self.seed()

        # We store the distance to the next pipe
        #self.next_pipe_dist = 99999
        
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # A bug in the PLE environement makes that if the player flap at two consecutive frame, the velocity of the player get back to 0
        # As this bug can be exploited by the agent to cheese the game, so each time the player flap, we determine that he should stay still on the next frame
        anti_cheese = False

        # Action number 0 is the flap
        if action == 0: anti_cheese = True

        # We transform our 0/1 action into an action the PLE environement will understand
        action = self.game_instance.getActionSet()[action]

        # We take the action and get the reward from the environement
        reward = self.game_instance.act(action) * 10

        # If we just flapped, we stay still for the next frame
        if anti_cheese: reward += self.game_instance.act(None) * 10

        # If the bird touches anything, the game is over
        if self.game_instance.game_over():
            done = True
        else:
            done = False
        
        # If the distance to the next pipe is higher than the precedent one it means we passed a pipe
        #if self._get_obs(override_type=VECTOR_STATE)[2] > self.next_pipe_dist: reward += 100; print("he made it !!!! YEEEESSS\n\n\n\n\n\n\n\n\n\nYESSSSSSSSSSSSSSSSSSSSS")
        
        # We update the distance to the next pipe accordingly
        #self.next_pipe_dist = self._get_obs(override_type=VECTOR_STATE)[2]
        #if done: self.next_pipe_dist = 9999
        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.env_state == IMAGE_STATE:
            image = self.game_instance.getScreenRGB()
            image = cv2.resize(image, dsize=(64, 36), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.array(image)
        if self.env_state == VECTOR_STATE:
            return np.array(list(self.game_instance.getGameState().values()))

    def reset(self):
        self.game_instance.reset_game()
        return self._get_obs()

    def render(self, mode='human', close=False):
        image = self.game_instance.getScreenRGB()
        return np.flip(np.rot90(image, k=3), axis=1)