import pandas as pd
import numpy as np
from build_surgery_metrics import run_supabase_query

class MetricBase:
    """
    Base class for time series metrics.

    Attributes:
        surgery_list_size (float): The size of the surgery list. This value can impact normalization.
        timeseries (pd.Series): The absolute time series data for the metric.
    """
    def __init__(self, surgery_list_size: float, timeseries: pd.Series):
        self.surgery_list_size = surgery_list_size
        self.timeseries = timeseries

    def get_absolute(self) -> pd.Series:
        """
        Returns the absolute time series data.
        """
        return self.timeseries

    def get_normalized(self) -> pd.Series:
        """
        Returns the normalized time series data.
        Default normalization divides the absolute values by the surgery list size.
        Subclasses can override this method for alternate normalization schemes.
        """
        # Avoid division by zero
        if self.surgery_list_size == 0:
            return self.timeseries
        return self.timeseries / self.surgery_list_size

    def display(self):
        """
        Display both the absolute and normalized time series.
        """
        print("Absolute values:")
        print(self.get_absolute())
        print("\nNormalized values:")
        print(self.get_normalized())


class CxScreeningMetric(MetricBase):
    """
    Concrete metric for cervical screening.

    For cx_screening, normalization might be more appropriately done using the monthly demand
    rather than the overall surgery list size.
    """
    def __init__(self, surgery_list_size: float, timeseries: pd.Series, monthly_demand: float):
        super().__init__(surgery_list_size, timeseries)
        self.monthly_demand = monthly_demand

    def get_normalized(self) -> pd.Series:
        """
        Normalizes the time series data by the monthly demand.
        """
        if self.monthly_demand == 0:
            return self.timeseries
        return self.timeseries / self.monthly_demand


def df_to_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'M', agg_func: str = 'sum') -> pd.Series:
    """
    Convert a pandas DataFrame to a time series.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        date_col (str): The name of the column containing date information.
        value_col (str): The name of the column containing the metric values.
        freq (str): The resampling frequency (default is 'M' for monthly).
        agg_func (str): The aggregation function to apply during resampling (default is 'sum').

    Returns:
        pd.Series: A time series with dates as index and aggregated metric values.

    The function:
    1. Converts the date_col to datetime.
    2. Sets the date_col as the DataFrame index.
    3. Sorts the index.
    4. Resamples the data to the specified frequency using the given aggregation function.
    """
    # Convert the date column to datetime if not already in datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Drop rows where date conversion failed
    df = df.dropna(subset=[date_col])

    # Set the date column as index and sort the DataFrame
    ts_df = df.set_index(date_col).sort_index()

    # Resample the DataFrame to the desired frequency and aggregate the value column
    if agg_func == 'sum':
        ts = ts_df[value_col].resample(freq).sum()
    elif agg_func == 'mean':
        ts = ts_df[value_col].resample(freq).mean()
    else:
        ts = ts_df[value_col].resample(freq).agg(agg_func)

    # Ensure the result is a Series (in case aggregation returns a DataFrame)
    if isinstance(ts, pd.DataFrame):
        ts = ts.iloc[:, 0]

    return ts


def import_supabase_data():
    """
    Import data from the Supabase database for cx_screening metrics.

    Queries:
        - cx_screening: contains id, event_date, pt_id, surgery_id.
        - surgery: contains id, name, list_size.
        - female_pts: contains id, pt_id, age, surgery_id.

    Processes:
        a) Computes overall monthly aggregated counts from the cx_screening table.
        b) Computes monthly aggregated counts per individual surgery (grouped by surgery_id).
        c) Calculates the monthly demand for cx_screening per surgery based on the eligible female patients.
           - Eligible patients are those aged between 25 and 65.
           - For patients aged 25-49: assumed to need 1 smear every 3 years (i.e., 5/3 smears in 5 years).
           - For patients aged 50-65: assumed to need 1 smear every 5 years.
           - Total smears over 5 years = (number in older group) + (5/3 * number in younger group).
           - Monthly demand = (total smears over 5 years) / 60.

    Returns:
        dict: {
            "cx_screening_overall": pd.Series of overall monthly counts,
            "cx_screening_per_surgery": dict mapping surgery_id to monthly pd.Series,
            "monthly_demand": dict mapping surgery_id to monthly demand (float),
            "surgery": pd.DataFrame of surgery data,
            "female_pts": pd.DataFrame of female patient data
        }
    """
    # Query the relevant tables
    query_cx = "SELECT id, event_date, pt_id, surgery_id FROM cx_screening;"
    query_surgery = "SELECT id, name, list_size FROM surgery;"
    query_female = "SELECT id, pt_id, age, surgery_id FROM female_pts;"

    df_cx = run_supabase_query(query_cx)
    df_surgery = run_supabase_query(query_surgery)
    df_female = run_supabase_query(query_female)

    # Ensure the dataframes are not None, assign empty DataFrames if necessary
    if df_cx is None:
        df_cx = pd.DataFrame(columns=["id", "event_date", "pt_id", "surgery_id"])
    if df_surgery is None:
        df_surgery = pd.DataFrame(columns=["id", "name", "list_size"])
    if df_female is None:
        df_female = pd.DataFrame(columns=["id", "pt_id", "age", "surgery_id"])

    # Overall monthly aggregate for cx_screening (count of tests per month)
    overall_ts = df_to_timeseries(df_cx, date_col='event_date', value_col='id', agg_func='count')

    # Monthly aggregate per surgery for cx_screening
    cx_per_surgery = {}
    for surgery_id, group in df_cx.groupby("surgery_id"):
        cx_per_surgery[surgery_id] = df_to_timeseries(group, date_col='event_date', value_col='id', agg_func='count')

    # Calculate monthly demand from female_pts table for each surgery
    monthly_demand = {}
    for surgery_id, group in df_female.groupby("surgery_id"):
        # Filter eligible patients: age between 25 and 65
        eligible = group[(group["age"] >= 25) & (group["age"] <= 65)]
        # Younger: age < 50, Older: age >= 50
        n_young = eligible[eligible["age"] < 50].shape[0]
        n_old = eligible[eligible["age"] >= 50].shape[0]
        # Total smears over 5 years:
        total_smears_5yr = n_old + (5/3) * n_young
        # Monthly demand:
        monthly = total_smears_5yr / 60.0  # 60 months in 5 years
        monthly_demand[surgery_id] = monthly

    return {
        "cx_screening_overall": overall_ts,
        "cx_screening_per_surgery": cx_per_surgery,
        "monthly_demand": monthly_demand,
        "surgery": df_surgery,
    }


# Example usage demonstrating the import of data and calculation of aggregates and demand.
if __name__ == "__main__":
    data = import_supabase_data()

    print("Overall Monthly Aggregated cx_screening Counts:")
    print(data["cx_screening_overall"])

    print("\nMonthly Aggregated cx_screening Counts per Surgery:")
    for surgery_id, ts in data["cx_screening_per_surgery"].items():
        print(f"Surgery {surgery_id}:")
        print(ts)

    print("\nMonthly Demand for cx_screening per Surgery:")
    for surgery_id, demand in data["monthly_demand"].items():
        print(f"Surgery {surgery_id}: {demand:.2f} tests per month")

    print("\nSurgery Data:")
    print(data["surgery"].head())
