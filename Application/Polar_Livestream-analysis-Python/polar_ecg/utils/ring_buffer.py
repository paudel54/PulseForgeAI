"""
Pre-allocated numpy ring buffer for fast plot updates.
Avoids deque -> list -> np.array copies on every frame.
"""

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer backed by a contiguous numpy array."""

    __slots__ = ("_buf", "_capacity", "_write_pos", "_count")

    def __init__(self, capacity: int, dtype=np.float64):
        self._buf = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def append(self, value: float):
        self._buf[self._write_pos] = value
        self._write_pos = (self._write_pos + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def extend(self, values):
        arr = np.asarray(values, dtype=self._buf.dtype)
        n = len(arr)
        if n == 0:
            return

        if n >= self._capacity:
            arr = arr[-self._capacity:]
            n = self._capacity
            self._buf[:] = arr
            self._write_pos = 0
            self._count = self._capacity
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = arr
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos:] = arr[:first]
            self._buf[: n - first] = arr[first:]

        self._write_pos = end % self._capacity
        self._count = min(self._count + n, self._capacity)

    def get_last_n(self, n: int) -> np.ndarray:
        n = min(n, self._count)
        if n == 0:
            return np.empty(0, dtype=self._buf.dtype)

        start = (self._write_pos - n) % self._capacity

        if start < self._write_pos:
            return self._buf[start:self._write_pos].copy()
        return np.concatenate((
            self._buf[start:],
            self._buf[:self._write_pos],
        ))
