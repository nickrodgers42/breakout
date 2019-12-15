import numpy as np
import random

class ReplayMemory():
    def __init__(self, size=1000000, frame_height=84, frame_width=84, frame_buff_size=4, batch_size=32):
        self._max_size = size
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._frame_buff_size = frame_buff_size
        self._batch_size = batch_size

        self._frames = np.empty((self._max_size, self._frame_height, self._frame_width), dtype=np.uint8)
        self._actions = np.empty(self._max_size, np.int32)
        self._rewards = np.empty(self._max_size, np.float32)
        self._gameovers = np.empty(self._max_size, np.bool)
        
        self._stored_memories = 0
        self._current = 0
        
    def add_memory(self, memory):
        frame, action, reward, gameover = memory
        frame = np.reshape(frame, (self._frame_height, self._frame_width))
        self._frames[self._current] = frame
        self._actions[self._current] = action
        self._rewards[self._current] = reward
        self._gameovers[self._current] = gameover
        
        self._current += 1
        self._stored_memories = max(self._stored_memories, self._current)

        self._current %= self._max_size

    def _get_valid_indices(self, num_indices=None):
        if num_indices is None:
            num_indices = self._batch_size
        num_indices = min(self.size, num_indices)

        valid_indices = []
        invalid_indices = []

        while len(valid_indices) < num_indices and len(invalid_indices) < self.size:
            # choose a random number between 0 and size
            start = random.randint(0, self.size - 1)
            end = start + self._frame_buff_size

            if end >= self.size and self.size < self._max_size:
                invalid_indices.append(start)
                continue
            
            if self._current >= start and self._current <= end % self._max_size:
                invalid_indices.append(start)
                continue

            range_valid = True
            for i in range(start, end):
                circularIndex = i % self._max_size
                if self._gameovers[circularIndex]:
                    range_valid = False
                    invalid_indices.append(start) if start not in invalid_indices else None
                    break
            if range_valid:
                valid_indices.append((end % self._max_size) - 1)

        return valid_indices

    def _get_frame_stack(self, index):
        start = index - self._frame_buff_size + 1
        frame_stack = []
        for i in range(start, start + self._frame_buff_size):
            circularIndex = i % self._max_size
            print(f'Index: {circularIndex}')
            frame_stack.append(self._frames[circularIndex])
        print(len(frame_stack))
        frame_stack = np.dstack(frame_stack)
        return frame_stack

    def sample(self, num_samples=None):
        if num_samples is None:
            num_samples = self._batch_size
        
        valid_indices = self._get_valid_indices(num_samples)

        memories = []
        for i in valid_indices:
            state = self._get_frame_stack(i)
            action = self._actions[i]
            reward = self._rewards[i]
            next_frame = self._frames[i + 1]
            memories.append((state, action, reward, next_frame))
        
        return memories

    @property
    def size(self):
        return self._stored_memories