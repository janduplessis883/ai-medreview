import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables once
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

def run_supabase_query(sql_query):
    try:
        # Connect to the database
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )

        # Create a cursor and execute the query
        cursor = connection.cursor()
        cursor.execute(sql_query)

        # Fetch results and column names
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(result, columns=columns)

        # Clean up
        cursor.close()
        connection.close()

        return df

    except Exception as e:
        print(f"Query failed: {e}")
        return None


if __name__ == "__main__":
    # Query to fetch surgery metrics
    sql_query = """
    SELECT
        *
    FROM
        appointments;
    """

    # Run the query
    df = run_supabase_query(sql_query)
    print(df.head())
