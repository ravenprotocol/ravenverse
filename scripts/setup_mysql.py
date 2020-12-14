import contextlib

import sqlalchemy

from common.constants import MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE


def delete_create_database():
    with sqlalchemy.create_engine('mysql://{}:{}@{}:{}/{}'.format(MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, "mysql"),
                                  isolation_level='AUTOCOMMIT').connect() as connection:
        with contextlib.suppress(sqlalchemy.exc.ProgrammingError):
            connection.execute("DROP DATABASE {}".format(MYSQL_DATABASE))
            print("Database deleted")
            connection.execute("CREATE DATABASE {}".format(MYSQL_DATABASE))
            print("Database created")
