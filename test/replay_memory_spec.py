import unittest
import random
import gym
import sys
import cv2
import numpy as np

sys.path.insert(1, '../')
from replay_memory import ReplayMemory
from frame_processor import FrameProcessor

class TestReplayMemory(unittest.TestCase):
    
    def test_add_memory(self):
        mem = ReplayMemory(5,1,1,1)
        self.assertEqual(mem.size, 0)
        mem.add_memory((1, 1, 1, False))
        self.assertEqual(mem.size, 1)
        for i in range(5):
            mem.add_memory((1,1,1,False))
        self.assertEqual(mem.size, 5)
    
    def test_get_valid_indices(self):
        random.seed(0)
        mem = ReplayMemory(10, 1, 1, 2, 5)
        self.assertEqual(mem._get_valid_indices(), [])
        mem.add_memory((0, 0, 1, False))
        self.assertEqual(mem._get_valid_indices(), [])
        mem.add_memory((1, 0, 1, False))
        mem.add_memory((2, 0, 1, False))
        self.assertEqual(mem._get_valid_indices(1), [1])
        mem.add_memory((3, 0, 1, False))
        mem.add_memory((4, 0, 1, True))
        mem.add_memory((5, 0, 1, False))
        mem.add_memory((6, 0, 1, False))
        mem.add_memory((7, 0, 1, False))
        self.assertEqual(sorted(mem._get_valid_indices(4)), [2, 3, 3, 6])
        mem.add_memory((8, 0, 1, True))
        mem.add_memory((9, 0, 1, False))
        mem.add_memory((10, 0, 1, False))
        mem.add_memory((11, 0, 1, False))
        self.assertEqual(sorted(mem._get_valid_indices()), [0, 0, 6, 6, 7])

    def xtest_sample(self):
        mem = ReplayMemory(20, 84, 84, 4, 3)
        fp = FrameProcessor()
        env = gym.make('BreakoutDeterministic-v4')
        env.reset()
        while mem.size < 20:
            action = env.action_space.sample()
            state, reward, gameover, _  = env.step(action)
            state = fp.preprocess(state)
            mem.add_memory((state, action, reward, gameover))
        memories = mem.sample()
        for i in memories:
            state, action, reward, next_frame = i
            print(state.shape)
            splitFrames = np.split(state, 4, axis=2)
            for j in splitFrames:
                print(j.shape)
                j = cv2.cvtColor(j, cv2.COLOR_RGB2BGR)
                cv2.imshow('Frame', j)
                cv2.waitKey(0)