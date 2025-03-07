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
    """
    Execute a SQL query against the Supabase database and return results as a DataFrame.

    Args:
        sql_query (str): SQL query to execute

    Returns:
        pd.DataFrame or None: DataFrame containing query results or None if query fails
    """
    connection = None
    cursor = None

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

        # Fetch results
        result = cursor.fetchall()

        # Check if cursor.description is None (happens with queries that return no results)
        if cursor.description is None:
            return pd.DataFrame()  # Return empty DataFrame

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(result, columns=columns)

        return df

    except Exception as e:
        print(f"Query failed: {e}")
        return None

    finally:
        # Ensure resources are properly closed even if an exception occurs
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def columns_to_datetime(df):
    """
    Convert all columns ending with 'date' to datetime format.

    Args:
        df (pd.DataFrame): DataFrame containing date columns to convert

    Returns:
        pd.DataFrame: DataFrame with date columns converted to datetime format

    Note:
        - Uses 'coerce' for errors, which converts invalid dates to NaT
        - Assumes date format is '%Y-%m-%d'
    """
    if df is None or df.empty:
        return df

    try:
        # Find all columns that end with 'date' (case insensitive)
        date_columns = [col for col in df.columns if col.lower().endswith('date')]

        # Convert each date column to datetime
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

            # Log columns with many NaT values (potential format issues)
            nat_count = df[col].isna().sum()
            if nat_count > 0 and nat_count > 0.1 * len(df):  # More than 10% NaT
                print(f"Warning: Column '{col}' has {nat_count} ({nat_count/len(df):.1%}) NaT values after conversion")

    except Exception as e:
        print(f"Error converting date columns: {e}")

    return df


def pre_process_dfs(dfs_dict):
    """
    Pre-process DataFrames to clean and standardize data for analysis.

    This function performs several operations on the appointments DataFrame:
    1. Removes surgery names from rota_type
    2. Filters out unwanted rota types and appointment statuses
    3. Standardizes appointment status values
    4. Renames appointment_date to event_date for consistency

    Args:
        dfs_dict (dict): Dictionary containing DataFrames with surgery data

    Returns:
        dict: Dictionary with pre-processed DataFrames

    Raises:
        KeyError: If required DataFrames or columns are missing
    """
    try:
        # Validate required keys exist
        if 'appointments' not in dfs_dict or 'surgery' not in dfs_dict:
            raise KeyError("Missing required DataFrames: 'appointments' and/or 'surgery'")

        # Clean surgery names in rota_type
        # ---------------------------------
        # Get surgery names and create a version with spaces instead of hyphens
        surgery_list = dfs_dict['surgery']['name'].to_list()
        clean_surgery_list = [surgery_name.replace("-", " ") for surgery_name in surgery_list]

        # Create regex pattern to match any surgery name followed by optional space
        pattern = "|".join(re.escape(surgery) + r"\s?" for surgery in clean_surgery_list)

        # Remove surgery names from rota_type
        dfs_dict['appointments']['rota_type'] = dfs_dict['appointments']['rota_type'].str.replace(pattern, "", regex=True)

        # Filter appointments by rota type
        # --------------------------------
        # Define unwanted rota types
        rotas_to_drop = ['HCA', 'Session', 'PCN']

        # Create mask and filter DataFrame
        mask = dfs_dict['appointments']['rota_type'].isin(rotas_to_drop)
        dfs_dict['appointments'] = dfs_dict['appointments'][~mask]

        # Filter appointments by status
        # ----------------------------
        # Define unwanted appointment statuses
        statuses_to_drop = ['Cancelled by Patient', 'Booked', 'Cancelled by Unit', 'Cancelled by Other Service']

        # Create mask and filter DataFrame
        mask = dfs_dict['appointments']['appointment_status'].isin(statuses_to_drop)
        dfs_dict['appointments'] = dfs_dict['appointments'][~mask]

        # Standardize appointment status values
        # ------------------------------------
        # Map 'Arrived' and 'In Progress' to 'Finished'
        dfs_dict['appointments'].loc[
            dfs_dict['appointments']['appointment_status'].isin(['Arrived', 'In Progress']),
            'appointment_status'
        ] = 'Finished'

        # Map 'Patient Walked Out' to 'Did Not Attend'
        dfs_dict['appointments'].loc[
            dfs_dict['appointments']['appointment_status'] == 'Patient Walked Out',
            'appointment_status'
        ] = 'Did Not Attend'

        # Rename columns for consistency
        # -----------------------------
        dfs_dict['appointments'].rename(columns={'appointment_date': 'event_date'}, inplace=True)

        return dfs_dict

    except Exception as e:
        print(f"Error in pre_process_dfs: {e}")
        # Return original dict to avoid breaking the pipeline
        return dfs_dict


def monthly_cxs_demand(df, surgery_id=2, verbose=False):
    """
    Calculate monthly cervical screening demand based on female patient demographics.

    Args:
        df (dict): Dictionary containing DataFrames with surgery data
        surgery_id (int, optional): ID of the surgery to analyze. Defaults to 2.
        verbose (bool, optional): Whether to print calculation details. Defaults to False.

    Returns:
        int: Estimated monthly cervical screening demand
    """
    # Filter female patients for the specified surgery
    surgery_female_pts = df['female_pts'][df['female_pts']['surgery_id'] == surgery_id]

    # Count patients by age group
    over_50 = len(surgery_female_pts[surgery_female_pts['age'] >= 50])
    under_50 = len(surgery_female_pts[surgery_female_pts['age'] < 50])  # Fixed: changed <= 50 to < 50

    # Calculate yearly demand based on screening frequency
    # Women over 50 are screened every 5 years, under 50 every 3 years
    woman_cx_per_year = over_50 / 5 + under_50 / 3
    monthly_demand = round(woman_cx_per_year / 12)

    # Print details if verbose mode is enabled
    if verbose:
        print(f"Surgery {surgery_id} - Yearly Cx Screening demand: {round(woman_cx_per_year)}")
        print(f"Monthly: {monthly_demand}")

    return monthly_demand


def one_surgery_df_dict(df_dict, surgery_id=2):
    """
    Filter data for a specific surgery and create derived appointment datasets.

    Args:
        df_dict (dict): Dictionary containing DataFrames with surgery data
        surgery_id (int, optional): ID of the surgery to analyze. Defaults to 2.

    Returns:
        dict: Dictionary containing filtered DataFrames for the specified surgery

    Raises:
        KeyError: If required DataFrames are missing from df_dict
        ValueError: If surgery_id is not found in the data
    """
    # Validate inputs
    required_keys = ["cx_screening", "bowel_normal", "bowel_non_responder", "bowel_positive", "appointments"]
    missing_keys = [key for key in required_keys if key not in df_dict]
    if missing_keys:
        raise KeyError(f"Missing required DataFrames: {', '.join(missing_keys)}")

    # Check if surgery_id exists in the data (if we can validate it)
    if ('surgery' in df_dict and
        'surgery_id' in df_dict['surgery'].columns and
        surgery_id not in df_dict['surgery']['surgery_id'].values):
        raise ValueError(f"Surgery ID {surgery_id} not found in the data")

    # Initialize result dictionary
    surgery_df_dict = {}

    # List of time series DataFrames to filter
    ts_list = ["cx_screening", "bowel_normal", "bowel_non_responder", "bowel_positive", "appointments"]

    # Filter each DataFrame for the specified surgery
    for ts_df in ts_list:
        # Initialize filtered_df to avoid "possibly unbound" errors
        filtered_df = pd.DataFrame()

        try:
            # Check if surgery_id column exists in this DataFrame
            if 'surgery_id' not in df_dict[ts_df].columns:
                print(f"Warning: 'surgery_id' column not found in {ts_df} DataFrame. Using all data.")
                filtered_df = df_dict[ts_df].copy()
            else:
                # Filter for the specified surgery_id
                filtered_df = df_dict[ts_df][df_dict[ts_df]['surgery_id'] == surgery_id]

                # Check if filtering resulted in empty DataFrame
                if filtered_df.empty:
                    print(f"Warning: No data found for surgery_id {surgery_id} in {ts_df} DataFrame.")

            # Store the filtered DataFrame
            surgery_df_dict[ts_df] = filtered_df

            # Create derived appointment DataFrames if processing appointments
            if ts_df == 'appointments' and not filtered_df.empty:
                if 'appointment_status' in filtered_df.columns:
                    # Appointments that were either finished or DNA
                    mask_used = (filtered_df['appointment_status'] == 'Finished') | (filtered_df['appointment_status'] == 'Did Not Attend')
                    surgery_df_dict[ts_df+"_used"] = filtered_df[mask_used]

                    # Appointments that were DNA (Did Not Attend)
                    surgery_df_dict[ts_df+"_dna"] = filtered_df[filtered_df['appointment_status'] == 'Did Not Attend']
                else:
                    print(f"Warning: 'appointment_status' column not found in {ts_df} DataFrame.")
                    surgery_df_dict[ts_df+"_used"] = pd.DataFrame()
                    surgery_df_dict[ts_df+"_dna"] = pd.DataFrame()
            elif ts_df == 'appointments':
                # Create empty DataFrames for appointments if filtered_df is empty
                surgery_df_dict[ts_df+"_used"] = pd.DataFrame()
                surgery_df_dict[ts_df+"_dna"] = pd.DataFrame()

        except Exception as e:
            print(f"Error filtering {ts_df} DataFrame: {e}")
            # Create an empty DataFrame to avoid downstream errors
            surgery_df_dict[ts_df] = pd.DataFrame()

            # Create empty derived DataFrames if this was the appointments DataFrame
            if ts_df == 'appointments':
                surgery_df_dict[ts_df+"_used"] = pd.DataFrame()
                surgery_df_dict[ts_df+"_dna"] = pd.DataFrame()

    return surgery_df_dict


def make_ts_df_dict(df_dict, surgery_id=2):
    """
    Create time series DataFrames by resampling data to monthly frequency.

    Args:
        df_dict (dict): Dictionary containing DataFrames with surgery data
        surgery_id (int, optional): ID of the surgery to analyze. Defaults to 2.

    Returns:
        dict: Dictionary containing monthly resampled time series DataFrames

    Raises:
        ValueError: If event_date column is missing or contains invalid dates
    """
    # Get filtered data for the specified surgery
    surgery_df_dict = one_surgery_df_dict(df_dict, surgery_id)

    # Define lists of DataFrames to process
    screening_list = ["cx_screening", "bowel_normal", "bowel_non_responder", "bowel_positive"]
    appointment_list = ["appointments_used", "appointments_dna"]

    # Process screening DataFrames
    for df_name in screening_list:
        try:
            # Skip processing if DataFrame is empty
            if surgery_df_dict[df_name].empty:
                print(f"Warning: {df_name} DataFrame is empty. Creating empty result DataFrame.")
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])
                continue

            # Check if event_date column exists
            if 'event_date' not in surgery_df_dict[df_name].columns:
                print(f"Warning: 'event_date' column missing in {df_name} DataFrame. Creating empty result DataFrame.")
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])
                continue

            # Get columns to drop - only drop if they exist
            cols_to_drop = [col for col in ['surgery_id', 'pt_id'] if col in surgery_df_dict[df_name].columns]

            # Set event_date as index and resample to monthly frequency
            surgery_df_dict[df_name] = (
                surgery_df_dict[df_name]
                .set_index('event_date')
                .resample('ME')
                .count()
            )

            # Drop columns if there are any to drop
            if cols_to_drop:
                surgery_df_dict[df_name] = surgery_df_dict[df_name].drop(columns=cols_to_drop)

            # If DataFrame is not empty, set the first column as 'count'
            if not surgery_df_dict[df_name].empty and len(surgery_df_dict[df_name].columns) > 0:
                surgery_df_dict[df_name] = surgery_df_dict[df_name].iloc[:, 0].to_frame()
                surgery_df_dict[df_name].columns = ['count']
            else:
                # Create empty DataFrame with count column
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])

        except Exception as e:
            print(f"Error processing {df_name}: {e}")
            # Create empty DataFrame with count column to avoid downstream errors
            surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])

    # Process appointment DataFrames
    for df_name in appointment_list:
        try:
            # Skip processing if DataFrame is empty
            if surgery_df_dict[df_name].empty:
                print(f"Warning: {df_name} DataFrame is empty. Creating empty result DataFrame.")
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])
                continue

            # Check if event_date column exists
            if 'event_date' not in surgery_df_dict[df_name].columns:
                print(f"Warning: 'event_date' column missing in {df_name} DataFrame. Creating empty result DataFrame.")
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])
                continue

            # Get columns to drop - only drop if they exist
            cols_to_drop = [col for col in ['surgery_id', 'pt_id', 'rota_type', 'appointment_status']
                           if col in surgery_df_dict[df_name].columns]

            # Set event_date as index and resample to monthly frequency
            surgery_df_dict[df_name] = (
                surgery_df_dict[df_name]
                .set_index('event_date')
                .resample('ME')
                .count()
            )

            # Drop columns if there are any to drop
            if cols_to_drop:
                surgery_df_dict[df_name] = surgery_df_dict[df_name].drop(columns=cols_to_drop)

            # If DataFrame is not empty, set the first column as 'count'
            if not surgery_df_dict[df_name].empty and len(surgery_df_dict[df_name].columns) > 0:
                surgery_df_dict[df_name] = surgery_df_dict[df_name].iloc[:, 0].to_frame()
                surgery_df_dict[df_name].columns = ['count']
            else:
                # Create empty DataFrame with count column
                surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])

        except Exception as e:
            print(f"Error processing {df_name}: {e}")
            # Create empty DataFrame with count column to avoid downstream errors
            surgery_df_dict[df_name] = pd.DataFrame(columns=['count'])

    return surgery_df_dict


def make_ts_dataframe(dfs_dict, surgery_id=2):
    """
    Create a heatmap of surgery metrics by processing and combining multiple time series.

    Args:
        dfs_dict (dict): Dictionary containing DataFrames with surgery data
        surgery_id (int, optional): ID of the surgery to analyze. Defaults to 2.

    Returns:
        pd.DataFrame: Combined DataFrame with processed metrics for heatmap visualization
    """
    # Get time series DataFrames dictionary
    ts_dict = make_ts_df_dict(dfs_dict, surgery_id)

    # Constants for calculations
    weeks_per_month = 52 / 12  # Average weeks per month

    # Get list_size dynamically from the surgery DataFrame
    try:
        if 'surgery' in dfs_dict and 'list_size' in dfs_dict['surgery'].columns:
            # Filter for the specified surgery_id
            surgery_row = dfs_dict['surgery'][dfs_dict['surgery']['id'] == surgery_id]

            if not surgery_row.empty and 'list_size' in surgery_row.columns:
                list_size = surgery_row['list_size'].iloc[0]
                print(f"Using dynamic list_size: {list_size} for surgery_id: {surgery_id}")
            else:
                # Fallback to default if surgery_id not found
                list_size = 4300
                print(f"Surgery ID {surgery_id} not found in surgery DataFrame. Using default list_size: {list_size}")
        else:
            # Fallback to default if surgery DataFrame or list_size column not available
            list_size = 4300
            print(f"Surgery DataFrame or list_size column not available. Using default list_size: {list_size}")
    except Exception as e:
        # Fallback to default if any error occurs
        list_size = 4300
        print(f"Error getting list_size: {e}. Using default list_size: {list_size}")

    try:
        # Process cx_screening - normalize by monthly demand
        monthly_demand = monthly_cxs_demand(dfs_dict, surgery_id=surgery_id)

        # Check if cx_screening DataFrame has data
        if not ts_dict['cx_screening'].empty and 'count' in ts_dict['cx_screening'].columns:
            ts_dict['cx_screening'] = (
                ts_dict['cx_screening']
                .assign(count=lambda df: df['count'] / monthly_demand)
                .fillna(0.0)
                .rename(columns={'count': 'cx_screening'})
            )
        else:
            # Create empty DataFrame with cx_screening column
            ts_dict['cx_screening'] = pd.DataFrame(columns=['cx_screening'])

        # Process bowel screening rate - calculate percentage of normal results
        # Check if all required bowel DataFrames have data
        bowel_dfs_valid = all(
            not ts_dict[df_name].empty and 'count' in ts_dict[df_name].columns
            for df_name in ['bowel_normal', 'bowel_non_responder', 'bowel_positive']
        )

        if bowel_dfs_valid:
            bowel_total = ts_dict['bowel_normal'] + ts_dict['bowel_non_responder'] + ts_dict['bowel_positive']
            ts_dict['bowel_screening_rate'] = (
                (ts_dict['bowel_normal'] / bowel_total)
                .fillna(0.0)
                .rename(columns={'count': 'bowel_screening_rate'})
            )
        else:
            # Create empty DataFrame with bowel_screening_rate column
            ts_dict['bowel_screening_rate'] = pd.DataFrame(columns=['bowel_screening_rate'])

        # Process appointments metrics with consistent approach
        for metric in ['appointments_used', 'appointments_dna']:
            try:
                # Check if metric DataFrame has data
                if not ts_dict[metric].empty and 'count' in ts_dict[metric].columns:
                    # Apply calculation to count column first
                    normalized_count = ts_dict[metric]['count'] / weeks_per_month / (list_size / 100) / 10

                    # Create a new DataFrame with the normalized values and handle NaN values
                    ts_dict[metric] = pd.DataFrame(
                        normalized_count,
                        index=ts_dict[metric].index
                    ).fillna(0.0)

                    # Rename the column
                    ts_dict[metric] = ts_dict[metric].rename(columns={0: metric})
                else:
                    # Create empty DataFrame with metric column
                    ts_dict[metric] = pd.DataFrame(columns=[metric])
            except Exception as e:
                print(f"Error processing {metric}: {e}")
                # Create empty DataFrame with metric column
                ts_dict[metric] = pd.DataFrame(columns=[metric])
    except Exception as e:
        print(f"Error in make_heatmap: {e}")
        # If any major error occurs, ensure all required DataFrames exist
        for metric in ['cx_screening', 'bowel_screening_rate', 'appointments_used', 'appointments_dna']:
            if metric not in ts_dict or ts_dict[metric].empty:
                ts_dict[metric] = pd.DataFrame(columns=[metric])

    # Define metrics to combine
    metrics_to_combine = ['cx_screening', 'bowel_screening_rate', 'appointments_used', 'appointments_dna']

    # Combine all DataFrames
    try:
        # Ensure all metrics exist before concatenating
        for metric in metrics_to_combine:
            if metric not in ts_dict:
                print(f"Warning: {metric} missing from ts_dict. Creating empty DataFrame.")
                ts_dict[metric] = pd.DataFrame(columns=[metric])

        # Create a list of DataFrames to concatenate
        dfs_to_concat = []
        for metric in metrics_to_combine:
            # Ensure the DataFrame has the correct column name
            if not ts_dict[metric].empty and len(ts_dict[metric].columns) > 0:
                if ts_dict[metric].columns[0] != metric:
                    ts_dict[metric].columns = [metric]
            dfs_to_concat.append(ts_dict[metric])

        # Concatenate DataFrames
        combined_df = pd.concat(dfs_to_concat, axis=1)

        # If combined_df is empty, create a DataFrame with all required columns
        if combined_df.empty:
            combined_df = pd.DataFrame(columns=metrics_to_combine)

    except Exception as e:
        print(f"Error combining DataFrames: {e}")
        # Create an empty DataFrame with all required columns as fallback
        combined_df = pd.DataFrame(columns=metrics_to_combine)

    return combined_df


def load_data():
    """
    Load all required data from the database and pre-process it for analysis.

    This function:
    1. Loads data from multiple tables in the database
    2. Converts date columns to datetime format
    3. Pre-processes the data for analysis

    Returns:
        dict: Dictionary containing all processed DataFrames ready for analysis

    Raises:
        RuntimeError: If critical data loading fails
    """
    # List of tables to load from the database
    table_list = [
        'appointments',
        'cx_screening',
        'surgery',
        'female_pts',
        'bowel_non_responder',
        'bowel_normal',
        'bowel_positive'
    ]

    # Dictionary to store DataFrames
    df_dict = {}
    failed_tables = []

    # Load each table
    for table in table_list:
        try:
            # Query to fetch all data from the table
            sql_query = f"""
            SELECT
                *
            FROM
                {table};
            """

            # Run the query and store result
            df = run_supabase_query(sql_query)

            # Check if query returned data
            if df is None or df.empty:
                print(f"Warning: No data returned for table '{table}'")
                failed_tables.append(table)
                continue

            # Convert date columns to datetime
            df = columns_to_datetime(df)

            # Store in dictionary and log shape
            df_dict[table] = df
            print(f"Table: {table} --------> DF Shape: {df_dict[table].shape}")

        except Exception as e:
            print(f"Error loading table '{table}': {e}")
            failed_tables.append(table)

    # Check if critical tables are missing
    critical_tables = ['appointments', 'surgery']
    missing_critical = [table for table in critical_tables if table in failed_tables]

    if missing_critical:
        raise RuntimeError(f"Failed to load critical tables: {', '.join(missing_critical)}")

    # Pre-process the data
    try:
        df_dict = pre_process_dfs(df_dict)
    except Exception as e:
        print(f"Warning: Error during pre-processing: {e}")
        print("Continuing with unprocessed data")

    return df_dict


def plot_heatmap(combined_df, surgery_id=2, figsize=(12, 8), cmap='YlOrRd', annot=True, title=None):
    """
    Plot a heatmap visualization of the surgery metrics.

    Args:
        combined_df (pd.DataFrame): DataFrame containing the metrics to visualize,
                                   typically the output of make_heatmap()
        surgery_id (int, optional): ID of the surgery being visualized. Defaults to 2.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (12, 8).
        cmap (str, optional): Colormap for the heatmap. Defaults to 'YlGnBu'.
        annot (bool, optional): Whether to annotate cells with values. Defaults to True.
        title (str, optional): Custom title for the plot. If None, a default title is used.

    Returns:
        matplotlib.figure.Figure: The figure object containing the heatmap

    Note:
        This function requires matplotlib and seaborn to be installed.
        Install with: pip install matplotlib seaborn
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
    except ImportError:
        print("Error: This function requires matplotlib and seaborn.")
        print("Install with: pip install matplotlib seaborn")
        return None

    # Check if combined_df is empty
    if combined_df.empty:
        print("Warning: Empty DataFrame provided. Cannot create heatmap.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available for heatmap",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        return fig

    # Create a copy to avoid modifying the original
    df = combined_df.copy()

    # Fill NaN values with 0 for visualization
    df = df.fillna(0)

    # Format the date index to 'Mon-YY' format (e.g., 'Jan-24')
    if isinstance(df.index, pd.DatetimeIndex):
        # Create a new DataFrame with formatted dates as index
        formatted_dates = df.index.strftime('%b-%y')  # %b gives abbreviated month name
        df.index = formatted_dates

    # Transpose the DataFrame to put months on the x-axis and metrics on the y-axis
    df = df.T

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap
    sns.heatmap(df, cmap=cmap, annot=annot, fmt='.2f', linewidths=0.5, ax=ax)

    # Set title
    if title is None:
        title = f"Surgery Metrics Heatmap (Surgery ID: {surgery_id})"
    ax.set_title(title, fontsize=14)

    # Format axis labels
    ax.set_xlabel('Month (MMM-YY)', fontsize=12)  # Updated to reflect abbreviated month format
    ax.set_ylabel('Metrics', fontsize=12)

    # Rotate x-axis labels to 90 degrees
    plt.xticks(rotation=90, ha='center')

    # Make y-axis labels horizontal
    plt.yticks(rotation=0)

    # Add colorbar label if it exists
    if len(ax.collections) > 0 and hasattr(ax.collections[0], 'colorbar'):
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label('Value', rotation=270, labelpad=20)

    # Improve layout
    plt.tight_layout()
    plt.show()




def save_heatmap(combined_df, output_path, surgery_id=2, **kwargs):
    """
    Create and save a heatmap visualization to a file.

    Args:
        combined_df (pd.DataFrame): DataFrame containing the metrics to visualize,
                                   typically the output of make_heatmap()
        output_path (str): Path where the heatmap image will be saved
        surgery_id (int, optional): ID of the surgery being visualized. Defaults to 2.
        **kwargs: Additional arguments to pass to plot_heatmap()

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: This function requires matplotlib.")
        print("Install with: pip install matplotlib")
        return False

    # Create the heatmap
    fig = plot_heatmap(combined_df, surgery_id=surgery_id, **kwargs)

    if fig is None:
        return False

    # Save the figure
    try:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Heatmap saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        return False


def plot_cx_screening_rate(df_dict, surgery_id=2, figsize=(10,6), title=None):
    """
    Plot actual cervical screening rates per month for the given surgery.

    This function uses the 'cx_screening' DataFrame from the monthly time series,
    computes the rate as the actual count divided by the estimated monthly demand,
    and plots the rate as a line chart.

    Args:
        df_dict (dict): Dictionary containing all processed DataFrames.
        surgery_id (int, optional): Surgery ID to analyze. Defaults to 2.
        figsize (tuple, optional): Figure size. Defaults to (10,6).
        title (str, optional): Custom title for the plot. If None, a default title is used.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    import matplotlib.pyplot as plt
    # Generate monthly time series data
    ts_dict = make_ts_df_dict(df_dict, surgery_id)
    if 'cx_screening' not in ts_dict or ts_dict['cx_screening'].empty:
        print("No cx_screening data available.")
        return None
    df = ts_dict['cx_screening'].copy()
    # Compute monthly demand
    monthly_demand = monthly_cxs_demand(df_dict, surgery_id)
    if monthly_demand == 0:
        print("Monthly demand is zero, cannot compute cx screening rate.")
        return None
    # Compute rate as actual count divided by demand
    df['cx_rate'] = df['count'] / monthly_demand
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df['cx_rate'], marker='o', linestyle='-')
    ax.set_xlabel('Month')
    ax.set_ylabel('Cervical Screening Rate')
    if title is None:
        title = f"Cervical Screening Rate Per Month (Surgery ID: {surgery_id})"
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    df_dict = load_data()

    # Create heatmap data
    combined_df = make_heatmap(df_dict, surgery_id=2)

    # Plot and save the heatmap
    save_heatmap(combined_df, "reports/surgery_metrics_heatmap.png",
                 figsize=(14, 10), cmap="YlOrBr", annot=True,
                 title="Surgery Metrics Monthly Heatmap")

    # Plot and save the cervical screening rates
    fig_cs = plot_cx_screening_rate(df_dict, surgery_id=2)
    if fig_cs:
        fig_cs.savefig("reports/cx_screening_rate.png", dpi=300, bbox_inches="tight")
        print("Cx screening rate plot saved to: reports/cx_screening_rate.png")
