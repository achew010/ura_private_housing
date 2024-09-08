import psycopg2
from psycopg2 import sql
import os
from db_constants import *
import time

class DatabaseManager:
    def __init__(self, database_name:str="uradb"):
        self.db_url = os.getenv('DATABASE_URL')
        self.conn = None

        while not self.conn:
            print("Connecting to database...")
            try:
                self.conn = psycopg2.connect(self.db_url)
            except:
                time.sleep(5)

    def close(self):
        self.conn.close()

    def save_data_to_db(self, data):
        """
        Save the data from the API to the PostgreSQL database.
        """

        try:
            # Connect to the PostgreSQL database
            cur = self.conn.cursor()
            
            # Insert API data into a table
            count = 0
            for item in data['Result']:
                transactions = item.pop('transaction')
                if len(item) == len(PROPERTY_FIELDS):
                    cur.execute(INSERT_PROPERTY_QUERY, tuple(item.values()))
                    property_id = cur.fetchone()[0]

                for transaction in transactions:
                    _query = INSERT_TRANSACTION_QUERY.format(property_id=property_id)
            
                    if len(transaction) == len(TRANSACTION_FIELDS):
                        cur.execute(_query, tuple(transaction.values()))
                        count += 1   
                    
            # Commit the transaction and close the connection
            self.conn.commit()
            cur.close()
            print(f"{count} transactions successfully saved to the database.")
        except Exception as e:
            print(f"Error saving data to database: {e}")

    def is_table_populated(self):
        """
        Checks if a specific database table is populated (i.e., contains any rows).
        
        Parameters:
            conn (psycopg2 connection): The database connection object.
        
        Returns:
            bool: True if the table is populated, False otherwise.
        """
        try:
            # Create a cursor object using the connection
            cur = self.conn.cursor()

            # Formulate the SQL query to count rows in the table
            query = sql.SQL("SELECT EXISTS (SELECT 1 FROM Transactions LIMIT 1)")

            # Execute the query
            cur.execute(query)

            # Fetch the result (True if rows exist, False otherwise)
            result = cur.fetchone()[0]

            cur.close()
            # Return the result
            return result

        except Exception as e:
            print(f"Error checking table {table_name}: {e}")
            return False


    def load_data_from_db(self):
        """
        Load data from the PostgreSQL database into a pandas DataFrame.
        """
        import pandas as pd
        try:
            transactions_data = pd.read_sql(LOAD_QUERY, self.conn)       
            return transactions_data
        except Exception as e:
            print(f"Error loading data from database: {e}")
            return None
