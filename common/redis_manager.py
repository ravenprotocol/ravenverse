import redis
from .constants import REDIS_HOST, REDIS_PORT, REDIS_DB
from .utils import Singleton


@Singleton
class RedisManager(object):
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def connect(self):
        return self.r


class RavQueue(object):
    def __init__(self, name):
        self.queue_name = name
        redis_manager = RedisManager.Instance()
        self.r = redis_manager.connect()

    def push(self, value):
        if self.search(value) == -1:
            return self.r.rpush(self.queue_name, value)
        else:
            return -1

    def pop(self):
        return self.r.lpop(self.queue_name)

    def __len__(self):
        return self.r.llen(self.queue_name)

    def remove(self, value):
        self.r.lrem(self.queue_name, count=0, value=value)

    def delete(self):
        return self.r.delete(self.queue_name)

    def get(self, index):
        return self.r.lindex(self.queue_name, index)

    def set(self, index, value):
        return self.r.lset(self.queue_name, index, value)

    def search(self, value):
        if type(value).__name__ != "str":
            value = str(value)
        elements = self.r.lrange(self.queue_name, 0, -1)
        try:
            return elements.index(bytes(value, "utf-8"))
        except ValueError as e:
            return -1


