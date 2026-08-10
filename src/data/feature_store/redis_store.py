import redis
import json


class RedisFeatureStore:

    def __init__(self):
        self.client = redis.Redis(host="localhost", port=6379)

    def set_features(self, symbol, features):
        self.client.set(symbol, json.dumps(features))

    def get_features(self, symbol):
        data = self.client.get(symbol)

        if data:
            return json.loads(data)

        return None