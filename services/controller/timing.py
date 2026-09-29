"""Monotonic wall-time accounting; interrupted operations retain elapsed time."""
from contextlib import contextmanager
import time


class Timings:
    def __init__(self):
        self.seconds = {}
        self.calls = {}

    @contextmanager
    def measure(self, name):
        started = time.monotonic()
        self.calls[name] = self.calls.get(name, 0) + 1
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.monotonic() - started

    def total(self):
        return sum(self.seconds.values())

    def snapshot(self):
        return {name: {'seconds': seconds, 'calls': self.calls[name]}
                for name, seconds in self.seconds.items()}
