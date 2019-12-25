from gym.envs.registration import register
 
register(id='FlappyB-v0', 
    entry_point='flappy_env.envs:FlappyEnv', 
)