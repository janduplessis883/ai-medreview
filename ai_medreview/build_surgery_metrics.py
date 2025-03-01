import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import re

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


def columns_to_datetime(df):
    date_columns = [col for col in df.columns if col.lower().endswith('date')]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
    return df


def pre_process_dfs(dfs_dict):

    # Fixes appointments dataframe
    surgery_list = dfs_dict['surgery']['name'].to_list()
    new_surgery_list = []
    for surgery_name in surgery_list:

        new_surgery_list.append(surgery_name.replace("-", " "))

    pattern = "|".join(re.escape(surgery) + r"\s?" for surgery in new_surgery_list)
    dfs_dict['appointments']['rota_type'] = dfs_dict['appointments']['rota_type'].str.replace(pattern, "", regex=True)

    # Drop unwaanted rota types with boolean mask
    rotas_to_drop = ['HCA', 'Session', 'PCN']
    mask = dfs_dict['appointments']['rota_type'].isin(rotas_to_drop)
    dfs_dict['appointments'] = dfs_dict['appointments'][~mask]

    list_to_drop = ['Cancelled by Patient', 'Booked', 'Cancelled by Unit', 'Cancelled by Other Service']

    mask = dfs_dict['appointments']['appointment_status'].isin(list_to_drop)
    dfs_dict['appointments'] = dfs_dict['appointments'][~mask]

    # Map appointment status to Finished
    dfs_dict['appointments'].loc[
        dfs_dict['appointments']['appointment_status'].isin(['Arrived', 'In Progress']),
        'appointment_status'
    ] = 'Finished'

    dfs_dict['appointments'].loc[
        dfs_dict['appointments']['appointment_status'] == 'Patient Walked Out',
        'appointment_status'
    ] = 'Did Not Attend'

    return dfs_dict


def monthly_cxs_demand(df, surgery_id=2):

    surgery_female_pts = df['female_pts'][df['female_pts']['surgery_id'] == surgery_id]

    over_50 = len(surgery_female_pts[surgery_female_pts['age'] >= 50])
    under_50 = len(surgery_female_pts[surgery_female_pts['age'] <= 50])

    woman_cx_per_year = over_50 / 5 + under_50 / 3

    print(f"Surgery {surgery_id} - Yearly Cx Screening demand: {round(woman_cx_per_year)}")
    print(f"Monthly: {round(woman_cx_per_year/12)}")

    return round(woman_cx_per_year/12)


def one_surgery_df_dict(df_dict, surgery_id=2):
    surgery_df_dict = {}
    ts_list = ["cx_screening", "bowel_normal", "bowel_non_responder", "bowel_positive"]

    for ts_df in ts_list:
        surgery_df_dict[ts_df] = df_dict[ts_df][df_dict[ts_df]['surgery_id'] == surgery_id]

    return surgery_df_dict

def make_ts_df_dict(df_dict, surgery_id=2):
    surgery_df_dict = one_surgery_df_dict(df_dict, surgery_id)

    df_list = ["cx_screening", "bowel_normal", "bowel_non_responder", "bowel_positive"]

    for df_name in df_list:
        surgery_df_dict[df_name] = surgery_df_dict[df_name].set_index('event_date')
        surgery_df_dict[df_name] = surgery_df_dict[df_name].resample('ME').count().drop(columns=['surgery_id', 'pt_id'])
        surgery_df_dict[df_name].columns = ['count']
    return surgery_df_dict

def make_heatmap(dfs_dict, surgery_id=2):
    ts_dict = make_ts_df_dict(dfs_dict, surgery_id)

    ts_dict['cx_screening']['count'] = ts_dict['cx_screening']['count'] / monthly_cxs_demand(dfs_dict, surgery_id=surgery_id)

    ts_dict['bowl_screenig_rate'] = (ts_dict['bowel_normal'] / (ts_dict['bowel_normal'] + ts_dict['bowel_non_responder'] + ts_dict['bowel_positive'])).fillna(0.0)

    return ts_dict['bowl_screenig_rate']


if __name__ == "__main__":
    table_list = ['appointments', 'cx_screening', 'surgery', 'female_pts', 'bowel_non_responder', 'bowel_normal', 'bowel_positive']

    df = {}
    for table in table_list:
        # Query to fetch surgery metrics
        sql_query = f"""
        SELECT
            *
        FROM
            {table};
        """

        # Run the query
        df[table] = run_supabase_query(sql_query)
        df[table] = columns_to_datetime(df[table]) #convert to datetime if ends with date
        print(f"Table: {table} --------> DF Shape: {df[table].shape}")

    df = pre_process_dfs(df)
    print("value_counts for appointment_status and rota_types")
    print(df['appointments']['appointment_status'].value_counts())
    print(df['appointments']['rota_type'].value_counts())
