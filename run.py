import argparse
from scripts import setup_n_clean
from common import db


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--step', dest="step", help='Perform this step')
    args = parser.parse_args()

    if args.step == "clean":
        setup_n_clean.clean()
    elif args.step == "setup_database":
        db.create_tables()
