import pandas as pd
from datetime import datetime, date
from email_sender import send_email_with_attachment

from sheethelper import SheetHelper
sh = SheetHelper('https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit?gid=0#gid=0', 0, '/Users/janduplessis/code/janduplessis883/ai-medreview/secret/google_sheets_secret.json')

# ------------------------------------------
surgery_string = 'Kensington-Park-Medical-Centre'
recipient_email_address = "jan.duplessis@nhs.net"
# ------------------------------------------
def get_modern_html(month):
    return f"""
<div style="max-width: 480px; margin: 32px auto; border: 2px solid #e0e0e0; border-radius: 18px; padding: 32px 24px; background: #fafbfc; font-family: 'Segoe UI', Arial, sans-serif;">
  <p style="font-size: 1.1em; color: #222; margin-bottom: 24px;">
    <strong>Dear AI MedReview User,</strong>
  </p>
  <p style="font-size: 1em; color: #333; margin-bottom: 24px;">
    Please find this automated email with your <b>FFT Extraction for month: {month}</b> attached as a CSV.
  </p>
  <p style="font-size: 1em; color: #555;">
    Regards,<br>
    <span style="font-weight: bold; color: #1976d2;">AI MedReview</span>
  </p>
</div>
"""


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
    from datetime import datetime

    # Always use previous month
    now = datetime.now()
    month = now.month - 1 if now.month > 1 else 12

    data = sh.gsheet_to_df()
    print(f"Data from Google Sheet: {data.shape}")

    surgery = process_data(data, 'Kensington-Park-Medical-Centre', month)
    print(f"Processed Data: {surgery.shape}")
    print(surgery.head())

    # Save processed data to CSV
    output_csv = "ai_medreview/data/processed_surgery.csv"
    surgery.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

    send_email_with_attachment(
        smtp_server="smtp.hostinger.com",
        smtp_port=587,
        smtp_user="hello@attribut.me",
        smtp_password="vuRmuxwyqge5vakwiz@",
        sender="hello@attribut.me",
        recipient=recipient_email_address,
        subject=f"AI MedReview FFT Extraction for Month: {month} - {surgery_string}",
        body=f"Dear AI MedReview {surgery_string},\n\nPlease find this automated email with your FFT Extraction for month: {month} attached as a csv.\n\nRegards,\nAI MedReview",
        attachment_path="ai_medreview/data/processed_surgery.csv",
        html_body=get_modern_html(month)
    )
    print("🎉 Email sent successfully with the processed data.")
