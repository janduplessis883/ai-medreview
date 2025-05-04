import pandas as pd
import os
import json
import resend # Added import
from loguru import logger # Added loguru import

# Configure loguru to write logs to a file in the 'logs' directory
logger.add("logs/app.log", rotation="1 MB", retention="30 days", level="INFO", mkdir=True)

resend.api_key = os.environ["RESEND_API_KEY"] # Added API key setup


# Assuming the Resend MCP tool is available and configured
# You might need to install the necessary library if using a direct API call or smtplib
# For MCP tool usage, the tool handles the underlying library.

# --- Configuration ---
# Path to your CSV data file.
# In GitHub Actions, you might need to adjust this path relative to the repository root.
# Or, if the file is fetched from an external source, add the fetching logic here.
CSV_FILE_PATH = 'ai_medreview/data/data.csv' # <-- **UPDATE THIS PATH**

# Define the criteria for identifying negative reviews.
SENTIMENT_COLUMNS = ['sentiment_free_text', 'sentiment_do_better']
SCORE_COLUMNS = ['sentiment_score_free_text', 'sentiment_score_do_better']
NEGATIVE_SENTIMENT_THRESHOLD = 0.6
TIME_COLUMN = 'time' # Column containing the timestamp/date
TIMEFRAME_HOURS = 24 # Timeframe in hours to check for recent reviews

# Email configuration
# These should ideally be stored as GitHub Secrets for security.
# RESEND_API_KEY = os.environ.get('RESEND_API_KEY') # <-- Get from GitHub Secrets
FROM_EMAIL = 'AI MedReview - FFT Alert <hello@attribut.me>' # <-- Fixed sender email as per user instructions
# TO_EMAILS_SECRET_PREFIX = 'email-' # Prefix for surgery email secrets (e.g., email-surgery-name)

# --- Function to identify negative reviews ---
def find_negative_reviews(df):
    """
    Identifies negative reviews based on sentiment columns and score threshold,
    filtered for the last X hours based on TIMEFRAME_HOURS.
    """
    if not all(col in df.columns for col in SENTIMENT_COLUMNS + SCORE_COLUMNS + [TIME_COLUMN]):
        missing_cols = [col for col in SENTIMENT_COLUMNS + SCORE_COLUMNS + [TIME_COLUMN] if col not in df.columns]
        logger.error(f"Missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame()

    # Convert 'time' column to datetime objects
    try:
        # Attempt to convert to datetime, coercing errors to NaT
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors='coerce')

        # Drop rows where datetime conversion failed
        initial_rows = len(df)
        df.dropna(subset=[TIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows due to invalid timestamps in '{TIME_COLUMN}'.")

        # Check if the DataFrame is empty after dropping NaNs
        if df.empty:
            logger.warning("DataFrame is empty after dropping rows with invalid timestamps.")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error processing '{TIME_COLUMN}' column for datetime conversion: {e}")
        return pd.DataFrame()

    # Calculate the time threshold based on the current time and timeframe
    time_threshold = pd.Timestamp.now() - pd.Timedelta(hours=TIMEFRAME_HOURS)

    # Filter for reviews within the specified timeframe
    recent_reviews_df = df[df[TIME_COLUMN] >= time_threshold].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Filter for negative sentiment with score >= threshold in either column
    negative_filtered_df = recent_reviews_df[
        (
            (recent_reviews_df[SENTIMENT_COLUMNS[0]].astype(str).str.lower() == 'negative') &
            (recent_reviews_df[SCORE_COLUMNS[0]] >= NEGATIVE_SENTIMENT_THRESHOLD)
        ) |
        (
            (recent_reviews_df[SENTIMENT_COLUMNS[1]].astype(str).str.lower() == 'negative') &
            (recent_reviews_df[SCORE_COLUMNS[1]] >= NEGATIVE_SENTIMENT_THRESHOLD)
        )
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    # Removed print(negative_filtered_df) - summary is printed below
    return negative_filtered_df

    # Filter for negative sentiment with score >= threshold in either column
    negative_filtered_df = recent_reviews_df[
        (
            (recent_reviews_df[SENTIMENT_COLUMNS[0]].str.lower() == 'negative') &
            (recent_reviews_df[SCORE_COLUMNS[0]] >= NEGATIVE_SENTIMENT_THRESHOLD)
        ) |
        (
            (recent_reviews_df[SENTIMENT_COLUMNS[1]].str.lower() == 'negative') &
            (recent_reviews_df[SCORE_COLUMNS[1]] >= NEGATIVE_SENTIMENT_THRESHOLD)
        )
    ].copy() # Use .copy() to avoid SettingWithCopyWarning


    return negative_filtered_df

# --- Function to format email content for a surgery ---
def format_email_content(surgery_name, negative_reviews_df):
    """
    Formats the email subject, plain text body, and HTML body for a specific surgery
    based on the negative reviews found.
    """
    num_reviews = len(negative_reviews_df)
    subject = f"AI MedReview: {num_reviews} New Negative Review(s) for {surgery_name}"

    # Plain text body
    text_body = f"The following new negative review(s) were found for {surgery_name} in the last 24 hours:\n\n"
    for index, row in negative_reviews_df.iterrows():
        text_body += f"Review Time: {row.get(TIME_COLUMN, 'N/A')}\n"
        text_body += f"General Feedback Sentiment: {row.get(SENTIMENT_COLUMNS[0], 'N/A')} (Score: {row.get(SCORE_COLUMNS[0], 'N/A')})\n"
        text_body += f"Improvement Suggestions Sentiment: {row.get(SENTIMENT_COLUMNS[1], 'N/A')} (Score: {row.get(SCORE_COLUMNS[1], 'N/A')})\n"
        text_body += f"Feedback: {row.get('free_text', 'N/A')}\n"
        text_body += f"Improvement Suggestion: {row.get('do_better', 'N/A')}\n"
        text_body += "-"*30 + "\n"
    text_body += "Regards,\nAI-MedReview Agent"


    # HTML body
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #f5f5f4;
      font-family: Arial, sans-serif;
    }}
    .email-container {{
      max-width: 600px;
      margin: auto;
      background-color: #ffffff;
      border: 1px solid #ccc;
      border-radius: 12px;
      padding: 24px;
    }}
    .review-item {{
        margin-bottom: 20px;
        padding-bottom: 20px;
        border-bottom: 1px solid #eee;
    }}
    .review-item:last-child {{
        border-bottom: none;
        padding-bottom: 0;
    }}
    strong {{
        color: #333;
    }}
    u {{
        color: #555;
    }}
    hr {{
        color: #e8e8e8;
    }}
  </style>
</head>
<body><BR><BR>
  <div class="email-container">
    <h1 style="color: #333;">{subject}</h1>
    <p style="color: #555;">
      The following new negative review(s) were found for <strong>{surgery_name}</strong> in the last 24 hours:
    </p><BR>
"""

    for index, row in negative_reviews_df.iterrows():
        html_body += f"""
    <div class="review-item">
        <p><u>Review Time: {row.get(TIME_COLUMN, 'N/A')}</u></p>
        <p><strong><u>Feedback Sentiment:</strong> {row.get(SENTIMENT_COLUMNS[0], 'N/A')} (Score: {row.get(SCORE_COLUMNS[0], 'N/A')})</u></p>
        <p><strong>Feedback</strong>: {row.get('free_text', 'N/A')}</p>
        <p><strong><u>Improvement Suggestions Sentiment:</strong> {row.get(SENTIMENT_COLUMNS[1], 'N/A')} (Score: {row.get(SCORE_COLUMNS[1], 'N/A')})</u></p>
        <p><strong>Improvement Suggestion</strong>: {row.get('do_better', 'N/A')}</p>
    </div>
"""

    html_body += """
    <p style="color: #555;">Regards,<br>AI-MedReview Agent</p><BR>
    <p style="font-size: 8pt; color: #888;">This ia an automated email sent from an AI-MedReview Agent using GitHub Actions, if you don't want to receive this anymore email Jan du Plessis.</p>
  </div>
<br><br></body>
</html>
"""

    return subject, text_body, html_body

# --- Function to send email using Resend Library ---
def send_alert_email(to_emails, subject, text_body, html_body):
    """
    Sends an email using the Resend Python library with both text and HTML bodies.
    """
    logger.info(f"Attempting to send email to: {to_emails}")
    logger.info(f"Subject: {subject}")
    # logger.debug(f"Text Body:\n{text_body}") # Use debug for potentially large bodies
    # logger.debug(f"HTML Body (partial):\n{html_body[:500]}...") # Use debug for potentially large bodies

    params: resend.Emails.SendParams = {
        "from": FROM_EMAIL, # Use the fixed FROM_EMAIL
        "to": [to_emails], # 'to' parameter expects a list of strings
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }

    try:
        email = resend.Emails.send(params)
        logger.info("Email sent successfully.")
        # logger.debug(f"Resend API response: {email}") # Use debug for API response
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        # Depending on requirements, you might want to raise the exception
        # or handle it differently (e.g., logging, retries).
        # For now, just logging the error.


# --- Main execution logic ---
if __name__ == "__main__":
    try:
        # Read the CSV file
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"Successfully read data from {CSV_FILE_PATH}. Shape: {df.shape}")

        # Find negative reviews
        negative_reviews = find_negative_reviews(df)
        logger.info(f"Found {len(negative_reviews)} negative reviews in the last {TIMEFRAME_HOURS} hours.")

        if not negative_reviews.empty:
            # Group by 'surgery' and send emails
            if 'surgery' in negative_reviews.columns:
                grouped_reviews = negative_reviews.groupby('surgery')

                for surgery_name, surgery_df in grouped_reviews:
                    # Construct the secret name for the email address
                    # Assuming surgery names are clean and can be used directly in secret names
                    # You might need to sanitize surgery_name if it contains special characters
                    email_secret_name = f"EMAIL_{surgery_name.replace('-', '_').replace(' ', '_').upper()}"

                    # Retrieve the email address from environment variables (simulating GitHub Secrets)
                    # In a real GitHub Action, you would access secrets directly.
                    # For this script, we'll simulate getting it from os.environ
                    to_email = os.environ.get(email_secret_name)

                    if to_email:
                        logger.info(f"Processing negative reviews for surgery: {surgery_name}")
                        subject, text_body, html_body = format_email_content(surgery_name, surgery_df)
                        send_alert_email(to_email, subject, text_body, html_body)
                    else:
                        logger.warning(f"Email address not found for surgery: {surgery_name} (looked for secret '{email_secret_name}')")
            else:
                logger.error("'surgery' column not found in the filtered negative reviews DataFrame.")
                # If 'surgery' column is missing, we can't group and send specific emails.
                # You might want to send a general alert or handle this case differently.
                # For now, we'll just log an error.

    except FileNotFoundError:
        logger.error(f"CSV file not found at {CSV_FILE_PATH}")
        exit(1) # Exit with error code
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(1) # Exit with error code
