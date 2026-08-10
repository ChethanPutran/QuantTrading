from collections import defaultdict


class Synchronizer:

    def __init__(self):
        self.windows = defaultdict(list)

    def add_event(self, timestamp, event):

        bucket = timestamp // 1000

        self.windows[bucket].append(event)

    def get_window(self, bucket):
        return self.windows[bucket]