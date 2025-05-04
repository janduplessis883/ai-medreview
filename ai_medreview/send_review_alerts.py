import pandas as pd
import os
import json
import resend # Added import

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
TIMEFRAME_HOURS = 300 # Timeframe in hours to check for recent reviews

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
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame()

    # Convert 'time' column to datetime objects
    try:
        # Attempt to convert to datetime, coercing errors to NaT
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors='coerce')

        # Drop rows where datetime conversion failed
        initial_rows = len(df)
        df.dropna(subset=[TIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to invalid timestamps in '{TIME_COLUMN}'.")

        # Check if the DataFrame is empty after dropping NaNs
        if df.empty:
            print("DataFrame is empty after dropping rows with invalid timestamps.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error processing '{TIME_COLUMN}' column for datetime conversion: {e}")
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
    print(negative_filtered_df)
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
    Formats the email subject and body for a specific surgery based on the negative reviews found.
    """
    if negative_reviews_df.empty:
        pass
    else:
        num_reviews = len(negative_reviews_df)
        subject = f"Action Required: {num_reviews} New Negative Review(s) for {surgery_name}"
        body = f"The following new negative review(s) were found for <h3>{surgery_name}</h3> in the last 24 hours:<BR><BR>"

        for index, row in negative_reviews_df.iterrows():
            # Include relevant sentiment information
            free_text_sentiment = row.get(SENTIMENT_COLUMNS[0], 'N/A')
            free_text_score = row.get(SCORE_COLUMNS[0], 'N/A')
            do_better_sentiment = row.get(SENTIMENT_COLUMNS[1], 'N/A')
            do_better_score = row.get(SCORE_COLUMNS[1], 'N/A')
            review_time = row.get(TIME_COLUMN, 'N/A')

            body += f"<u>Review Time: {review_time}</u><BR>"
            body += f"<strong>Feedback Sentiment: {free_text_sentiment} (Score: {free_text_score})</strong><BR>"
            body += f"<strong>Improvement Suggestions Sentiment: {do_better_sentiment} (Score: {do_better_score})</strong><BR>"
            # Assuming there's a column for the actual review text, e.g., 'review_text'
            # You might need to adjust this based on your actual data columns
            body += f"<strong>Feedback</strong>: {row.get('free_text', 'N/A')}<BR>"
            body += f"<strong>Improvement Suggestion</strong>: {row.get('do_better', 'N/A')}<BR>"
            body += "<hr><BR>"
        body += "Regards,\nAI-MedReview Agent"

    return subject, body

# --- Function to send email using Resend Library ---
def send_alert_email(to_emails, subject, body):
    """
    Sends an email using the Resend Python library.
    """
    print(f"Attempting to send email to: {to_emails}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")

    params: resend.Emails.SendParams = {
        "from": FROM_EMAIL, # Use the fixed FROM_EMAIL
        "to": [to_emails], # 'to' parameter expects a list of strings
        "subject": subject,
        "html": body, # Use 'text' for plain text body
    }

    try:
        email = resend.Emails.send(params)
        print("Email sent successfully:")
        print(email)
    except Exception as e:
        print(f"Error sending email: {e}")
        # Depending on requirements, you might want to raise the exception
        # or handle it differently (e.g., logging, retries).
        # For now, just printing the error.


# --- Main execution logic ---
if __name__ == "__main__":
    try:
        # Read the CSV file
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully read data from {CSV_FILE_PATH}. Shape: {df.shape}")

        # Find negative reviews
        negative_reviews = find_negative_reviews(df)
        print(f"Found {len(negative_reviews)} negative reviews in the last 24 hours.")

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
                        print(f"Processing negative reviews for surgery: {surgery_name}")
                        subject, body = format_email_content(surgery_name, surgery_df)
                        send_alert_email(to_email, subject, body)
                    else:
                        print(f"Warning: Email address not found for surgery: {surgery_name} (looked for secret '{email_secret_name}')")
            else:
                print("Error: 'surgery' column not found in the filtered negative reviews DataFrame.")
                # If 'surgery' column is missing, we can't group and send specific emails.
                # You might want to send a general alert or handle this case differently.
                # For now, we'll just print a warning.

    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        exit(1) # Exit with error code
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1) # Exit with error code
