#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    My sql database instance
"""

import MySQLdb


class GarmentDataModel:

    """
        Class DataModel will handle all data requests
        and automatically manage the connection of MySQL
    """

    # MySQL database instance
    mysql_db = None

    # Cursor instance for handling query
    cursor = None

    # Initialization, open the mysql
    def __init__(self, hostname, username, userpasswd, dbname):
        self.mysql_db = MySQLdb.connect(hostname, username, userpasswd, dbname)
        self.cursor = self.mysql_db.cursor()

    # Deconstructor, close the mysql connection
    def __del__(self):
        self.close_connection()

    # Close the connection
    def close_connection(self):
        self.mysql_db.close()

    # Execute a query
    def query(self, query_string):
        self.cursor.execute(query_string)
        data = self.cursor.fetchall()
        return data
