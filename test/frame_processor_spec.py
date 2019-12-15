import unittest
import cv2
import sys
sys.path.insert(1, '../')
from frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.fp = FrameProcessor()
        self.frame = cv2.imread('./test_frame.png')
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def test_init(self):
        self.assertIsNotNone(self.fp)
        self.assertEqual(self.fp._height, 84)
        self.assertEqual(self.fp._width, 84)

        fp = FrameProcessor(512, 256)
        self.assertEqual(fp._height, 512)
        self.assertEqual(fp._width, 256)
    
    def test_grayscale(self):
        newFrame = self.fp.grayscale(self.frame)
        self.assertEqual(len(self.frame.shape), 3)
        self.assertEqual(len(newFrame.shape), 2)
        self.assertEqual(self.frame.shape[0:2], newFrame.shape)

    def test_crop(self):
        newFrame = self.fp.crop(self.frame)
        self.assertLess(newFrame.shape[0], self.frame.shape[0])
        newFrame = self.fp.crop(self.frame, (50, 100, 50, 100))
        self.assertEqual(newFrame.shape[0], newFrame.shape[1])
    
    def test_resize(self):
        newFrame = self.fp.resize(self.frame)
        self.assertEqual(newFrame.shape, (84, 84, 3))
        newFrame = self.fp.resize(self.frame, (100, 100))
        self.assertEqual(newFrame.shape, (100, 100, 3))

    def test_preprocess(self):
        newFrame = self.fp.preprocess(self.frame)
        self.assertEqual(newFrame.shape, (84, 84))