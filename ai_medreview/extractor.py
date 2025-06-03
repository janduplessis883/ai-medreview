import pandas as pd
from datetime import datetime, date

from sheethelper import SheetHelper
sh = SheetHelper('https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit?gid=0#gid=0', 0, '/Users/janduplessis/code/janduplessis883/ai-medreview/secret/google_sheets_secret.json')


def process_data(df: pd.DataFrame, surgery_string: str, month: int) -> pd.DataFrame:
    # Example processing: filter out rows where 'column_name' is NaN
    df.columns = ['1', 'id1', 'date',
       'rating',
       'freetext',
       'do_better',
       'pcn', 'surgery', '3',
       '4',
       '5',
       '6']
    df.drop(columns=['1', '3', '4', '5', '6'], inplace=True)
    df.sort_values(by='date')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    single_surgery = df[df['surgery'] == surgery_string]
    single_surgery = single_surgery[(single_surgery['date'] >= datetime(2025, month, 1, 0, 0, 0))]

    return single_surgery


if __name__ == "__main__":
    data = sh.gsheet_to_df()
    print(f"Data from Google Sheet: {data.shape}")

    surgery = process_data(data, 'Kensington-Park-Medical-Centre', 5)
    print(f"Processed Data: {surgery.shape}")
    print(surgery.head())

    # Save processed data to CSV
    output_csv = "ai_medreview/data/processed_surgery.csv"
    surgery.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")
