import argparse
from scripts import setup_n_clean
from common import db, RavQueue
from scripts.setup_mysql import delete_create_database


def clear_redis_queues():
    QUEUE_HIGH_PRIORITY = "queue:high_priority"
    QUEUE_LOW_PRIORITY = "queue:low_priority"
    QUEUE_COMPUTING = "queue:computing"
    r = RavQueue(QUEUE_HIGH_PRIORITY)
    r.delete()
    r1 = RavQueue(QUEUE_LOW_PRIORITY)
    r1.delete()
    r2 = RavQueue(QUEUE_COMPUTING)
    r2.delete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--step', dest="step", help='Perform this step')
    args = parser.parse_args()

    if args.step == "clean":
        # Remove files
        setup_n_clean.clean()

        # Clear redis queues
        clear_redis_queues()
    elif args.step == "setup_database":
        delete_create_database()
        print("Creating tables")
        db.create_tables()
