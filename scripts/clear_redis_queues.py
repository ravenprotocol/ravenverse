from common import RavQueue

if __name__ == '__main__':
    QUEUE_HIGH_PRIORITY = "queue:high_priority"
    QUEUE_LOW_PRIORITY = "queue:low_priority"
    QUEUE_COMPUTING = "queue:computing"
    r = RavQueue(QUEUE_HIGH_PRIORITY)
    r.delete()
    r1 =  RavQueue(QUEUE_LOW_PRIORITY)
    r1.delete()
    r2 = RavQueue(QUEUE_COMPUTING)
    r2.delete()