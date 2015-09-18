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
        """
        Initialize the Data Model provider, start connection of MySQL

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. hostname          | String                 | host name for MySQL connection
        2. username          | String                 | MySQL database user name
        3. userpasswd        | String                 | MySQL database user password
        4. dbname            | String                 | Database that will be used in MySQL

        """

        self.mysql_db = MySQLdb.connect(hostname, username, userpasswd, dbname)
        self.cursor = self.mysql_db.cursor()

    # Deconstructor, close the mysql connection
    def __del__(self):

        """
        Deconstruct the instance and close the MySQL connection when finished.
        """

        self.close_connection()

    # Close the connection
    def close_connection(self):

        """
        Explicitly invoking when somebody want to close MySQL connection
        """

        self.mysql_db.close()

    # Execute a query
    def query(self, query_string):

        """
            Act a SQL query

            INPUT:
               PARAMS            | TYPE                   | DESCRIPTION
            1. hostname          | String                 | host name for MySQL connection

        """

        self.cursor.execute(query_string)
        data = self.cursor.fetchall()
        return data
