import cv2
import gym
import gym.spaces
import numpy as np
import collections

class FireResetEnv(gym.Wrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    For game of pong the environment requires a Fire action to
    be taken to start a new game. Also, to avoid slider at the 
    edge of the screen we adds checks on reset.

    Implementation taken from Open AI.
    '''
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'

        assert len(self.env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):

        self.env.reset()

        obs, _, done, _ = self.env.step(1)

        if done:
            self.env.reset()

        obs, _, done, _ = self.env.step(2)

        if done:
            self.reset()

        return obs

class MaxAndSkipEnv(gym.Wrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    We dont require to take action on every frame. We can safely
    skip some frames. More frames we skip the agents performance 
    decreases. Also, if we do not skip any frames training time 
    increases.
    
    Implementation taken from Open AI.
    '''
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done = None
        
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            
            self._obs_buffer.append(obs)
            
            total_reward += reward
            
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """
        Clear past frame buffer and init. to first obs. from inner env.
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    Routine resizes each frame to 84x84 pixels.
    '''
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # Normalise the image.
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        # don't need reward information on frame for learning remove it.
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    Pytorch CNN layer accepts images as CHANNEL x HEIGHT x WIDTH. 
    Convert incoming image shape from HEIGHT x WIDTH x CHANNEL to above shape.
    '''
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    Convert image values range from [0,255] to [0,1]
    '''
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    '''
    A class that is a wrapper around Open gym AI environment.
    To get the direction and velocity of the ball moving in the
    game we need to input a stack of frame. Alternative is to use
    a recurrent neural network. But, recurrent network takes time to
    train.
    '''
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

def make_env(env_name):
    env = gym.make(env_name)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env












