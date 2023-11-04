import mysql.connector

import mysql.connector

class MySQLDatabase:
    """
    A class representing a MySQL database connection.

    Attributes:
    -----------
    host : str
        The hostname or IP address of the MySQL server.
    user : str
        The MySQL user to authenticate as.
    password : str
        The password for the MySQL user.
    database : str
        The name of the database to use.
    port : int, optional
        The port number to use for the MySQL connection. Defaults to 3306.

    Methods:
    --------
    query(sql: str, params: tuple) -> list
        Executes a SQL query and returns the results as a list of tuples.
    insert_if_no_match(filename: str, user: str, content: str) -> bool
        Inserts a new record into the 'files' table if no matching record exists.
    insert_or_update(filename: str, user: str, content: str) -> None
        Inserts a new record into the 'files' table or updates an existing record.
    close() -> None
        Closes the database connection.
    """

    def __init__(self, host, user, password, database, port=3306):
        """
        Initializes a new MySQLDatabase instance.

        Parameters:
        -----------
        host : str
            The hostname or IP address of the MySQL server.
        user : str
            The MySQL user to authenticate as.
        password : str
            The password for the MySQL user.
        database : str
            The name of the database to use.
        port : int, optional
            The port number to use for the MySQL connection. Defaults to 3306.
        """
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port
        }
        self.conn = mysql.connector.connect(**self.config)
        self.cursor = self.conn.cursor()

    def query(self, sql, params=None):
        """
        Executes a SQL query and returns the results as a list of tuples.

        Parameters:
        -----------
        sql : str
            The SQL query to execute.
        params : tuple, optional
            A tuple of parameters to substitute into the SQL query.

        Returns:
        --------
        list
            A list of tuples representing the rows returned by the query.
        """
        self.cursor.execute(sql, params or ())
        return self.cursor.fetchall()

    def insert_if_no_match(self, filename, user, content):
        """
        Inserts a new record into the 'files' table if no matching record exists.

        Parameters:
        -----------
        filename : str
            The name of the file to insert.
        user : str
            The name of the user who owns the file.
        content : str
            The contents of the file.

        Returns:
        --------
        bool
            True if the record was inserted, False if a matching record already exists.
        """
        # Check if a record with the given filename and user already exists
        check_sql = "SELECT * FROM files WHERE filename = %s AND user = %s"
        self.cursor.execute(check_sql, (filename, user))
        
        if not self.cursor.fetchone():
            # If no match, insert the data
            insert_sql = "INSERT INTO files (filename, user, content) VALUES (%s, %s, %s)"
            self.cursor.execute(insert_sql, (filename, user, content))
            self.conn.commit()
            return True  # Indicates that the data was inserted
        return False  # Indicates that a match was found and no insertion occurred

    def insert_or_update(self, filename, user, content):
        """
        Inserts a new record into the 'files' table or updates an existing record.

        Parameters:
        -----------
        filename : str
            The name of the file to insert or update.
        user : str
            The name of the user who owns the file.
        content : str
            The contents of the file.

        Returns:
        --------
        None
        """
        sql = """
        INSERT INTO files (filename, user, content) 
        VALUES (%s, %s, %s) 
        ON DUPLICATE KEY UPDATE content = %s;
        """
        self.cursor.execute(sql, (filename, user, content, content))
        self.conn.commit()


    def close(self):
        """
        Closes the database connection.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.cursor.close()
        self.conn.close()

# # Usage
# db = MySQLDatabase(host='localhost', user='your_username', password='your_password', database='filedb')

# # Insert data if no match for filename and user
# result = db.insert_if_no_match('example.txt', 'JohnDoe', 'This is the content of the file.')
# if result:
#     print("Data inserted!")
# else:
#     print("Match found. No data inserted.")

# # Query data
# data = db.query("SELECT * FROM files")
# for record in data:
#     print(record)

# db.close()
