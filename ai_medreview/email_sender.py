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
    attachment_path: str,
    html_body: str = None
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
        body (str): Email body text (plain text fallback).
        attachment_path (str): Path to the CSV file to attach.
        html_body (str, optional): HTML body for the email.
    """
    print("🚀 Starting email sending process...")

    try:
        print("📝 Creating email message...")
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)
        if html_body:
            msg.add_alternative(html_body, subtype="html")
            print("🌐 HTML body added to email.")
        print("✅ Email message created successfully.")
    except Exception as e:
        print(f"❌ Error creating email message: {e}")
        raise

    # Attach the CSV file
    try:
        print(f"📎 Attaching file: {attachment_path}")
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
            print(f"✅ File '{file_name}' attached successfully.")
        else:
            print(f"❌ Attachment file not found: {attachment_path}")
            raise FileNotFoundError(f"Attachment file not found: {attachment_path}")
    except Exception as e:
        print(f"❌ Error attaching file: {e}")
        raise

    # Send the email via SMTP
    try:
        print(f"🔌 Connecting to SMTP server {smtp_server}:{smtp_port}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            print("🔒 Starting TLS encryption...")
            server.starttls()
            print("🔑 Logging in to SMTP server...")
            server.login(smtp_user, smtp_password)
            print("✉️ Sending email...")
            server.send_message(msg)
            print(f"✅ Email sent to {recipient} with attachment '{attachment_path}'")
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP authentication failed: {e}")
        raise
    except smtplib.SMTPConnectError as e:
        print(f"❌ SMTP connection failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        raise

if __name__ == "__main__":
    import sys

    # HTML template for the email body
    modern_html = """
    <div style="max-width: 480px; margin: 32px auto; border: 2px solid #e0e0e0; border-radius: 18px; padding: 32px 24px; background: #fafbfc; font-family: 'Segoe UI', Arial, sans-serif;">
      <p style="font-size: 1.1em; color: #222; margin-bottom: 24px;">
        <strong>Dear AI MedReview User,</strong>
      </p>
      <p style="font-size: 1em; color: #333; margin-bottom: 24px;">
        Please find this automated email with your <b>FFT Extraction for month: 5</b> attached as a CSV.
      </p>
      <p style="font-size: 1em; color: #555;">
        Regards,<br>
        <span style="font-weight: bold; color: #1976d2;">AI MedReview</span>
      </p>
    </div>
    """

    # Read environment variables
    required_env = [
        "SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SENDER", "RECIPIENT"
    ]
    missing = [var for var in required_env if var not in os.environ]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    smtp_server = os.environ["SMTP_SERVER"]
    smtp_port = int(os.environ["SMTP_PORT"])
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    sender = os.environ["SENDER"]
    recipient = os.environ["RECIPIENT"]

    subject = "Automated Email with Monthly Surgery Data Extraction"
    body = (
        "Dear AI MedReview User,\n\n"
        "Please find this automated email with your FFT Extraction for month: 5 attached as a csv.\n\n"
        "Regards,\nAI MedReview"
    )
    attachment_path = "ai_medreview/data/processed_surgery.csv"

    send_email_with_attachment(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        sender=sender,
        recipient=recipient,
        subject=subject,
        body=body,
        attachment_path=attachment_path,
        html_body=modern_html
    )
