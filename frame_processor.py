import cv2

class FrameProcessor():
    def __init__(self, height=84, width=84, gameArea=(30, 210, 0, 160)):
        self._height = height
        self._width = width
        self._gameArea = gameArea
    
    def grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    def crop(self, frame, gameArea=None):
        if gameArea is None:
            gameArea = self._gameArea
        return frame[gameArea[0]:gameArea[1], gameArea[2]:gameArea[3]]
    
    def resize(self, frame, size=None):
        if size is not None:
            return cv2.resize(frame, size)
        return cv2.resize(frame, (self._height, self._width))
    
    def preprocess(self, frame):
        frame = self.crop(frame)
        frame = self.grayscale(frame)
        frame = self.resize(frame)
        return frame