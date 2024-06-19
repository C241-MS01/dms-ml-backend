import mysql.connector


class Mysql:
    def __init__(self, host, user, password, database):
        config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
        }

        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="dms_pool", pool_size=10, **config
        )

    def get_connection(self):
        return self.pool.get_connection()
