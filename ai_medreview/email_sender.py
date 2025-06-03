import smtplib
from email.message import EmailMessage
import mimetypes
import os

def send_email_with_attachment(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str
):
    """
    Send an email with a CSV attachment using Python's built-in libraries.

    Args:
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port (e.g., 587 for TLS).
        smtp_user (str): SMTP username.
        smtp_password (str): SMTP password.
        sender (str): Sender email address.
        recipient (str): Recipient email address.
        subject (str): Email subject.
        body (str): Email body text.
        attachment_path (str): Path to the CSV file to attach.
    """
    # Create the email message
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the CSV file
    if attachment_path and os.path.isfile(attachment_path):
        ctype, encoding = mimetypes.guess_type(attachment_path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
            msg.add_attachment(
                file_data,
                maintype=maintype,
                subtype=subtype,
                filename=file_name
            )
    else:
        raise FileNotFoundError(f"Attachment file not found: {attachment_path}")

    # Send the email via SMTP
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        print(f"Email sent to {recipient} with attachment {attachment_path}")

# Example usage (uncomment and fill in your details to use):
send_email_with_attachment(
    smtp_server="smtp.hostinger.com",
    smtp_port=465,
    smtp_user="hello@attribut.me",
    smtp_password="vuRmuxwyqge5vakwiz@",
    sender="hello@attribut.me",
    recipient="drjanduplessis@icloud.com",
    subject="Automated Email with Monthly Surgery Data Extraction",
    body="Please find the attached CSV file with the processed surgery data.",
    attachment_path="ai_medreview/data/processed_surgery.csv"
)
