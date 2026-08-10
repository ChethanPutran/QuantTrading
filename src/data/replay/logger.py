import json


class ReplayLogger:

    def __init__(self, path):
        self.path = path

    def log(self, event):

        with open(self.path, "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")