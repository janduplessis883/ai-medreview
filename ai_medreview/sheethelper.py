import os
import pandas as pd
import gspread
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

secret_path = os.getenv("SECRET_PATH")


class SheetHelper:
    def __init__(self, sheet_url=None, sheet_id=0, secret_file_path=secret_path):
        self.sheet_instance = self.authenticate(sheet_url, sheet_id, secret_file_path)

    def authenticate(self, sheet_url, sheet_id, secret_file_path):
        credentials = service_account.Credentials.from_service_account_file(
            secret_file_path
        )
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = credentials.with_scopes(scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url)
        return sheet.get_worksheet(sheet_id)

    def append_row(self, row_list):
        self.sheet_instance.append_row(row_list)

    def get_last_row_index(self):
        return len(self.sheet_instance.get_all_records())

    def update_cell(self, row, col, value):
        self.sheet_instance.update_cell(row, col, value)

    def gsheet_to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.sheet_instance.get_all_records())
