import resend
import streamlit as st
import os

resend.api_key = st.secrets["RESEND_API_KEY"]

surgery_list = [
    'Earls-Court-Medical-Centre',
    'Earls-Court-Surgery',
    'Emperors-Gate-Health-Centre',
    'Health-Partners-at-Violet-Melchett',
    'Kensington-Park-Medical-Centre',
    'Knightsbridge-Medical-Centre',
    'Royal-Hospital-Chelsea',
    'Scarsdale-Medical-Centre',
    'Stanhope-Mews-Surgery',
    'The-Abingdon-Medical-Practice',
    'The-Chelsea-Practice',
    'The-Good-Practice'
]

surgery_info = {
    'Earls-Court-Medical-Centre': {'pm_name': 'Jalpa Patel', 'pm_email': 'jalpa.patel2@nhs.net'},
    'Earls-Court-Surgery': {'pm_name': 'Jan du Plessis', 'pm_email': 'jan.duplessis@nhs.net'},
    'Emperors-Gate-Health-Centre': {'pm_name': 'Esti Ballestero', 'pm_email': 'e.ballestero@nhs.net'},
    'Health-Partners-at-Violet-Melchett': {'pm_name': 'Warwick Young', 'pm_email': 'warwick.young@nhs.net'},
    'Kensington-Park-Medical-Centre': {'pm_name': 'Lesley French', 'pm_email': 'lesley.french@nhs.net'},
    'Knightsbridge-Medical-Centre': {'pm_name': 'Sue Neville', 'pm_email': 's.neville1@nhs.net'},
    'Royal-Hospital-Chelsea': {'pm_name': 'Amanda Lord', 'pm_email': 'amanda.lord@nhs.net'},
    'Scarsdale-Medical-Centre': {'pm_name': 'Marzena Grzymala', 'pm_email': 'marzena.grzymala@nhs.net'},
    'Stanhope-Mews-Surgery': {'pm_name': 'Sam Uddin', 'pm_email': 'sam.uddin@nhs.net'},
    'The-Abingdon-Medical-Practice': {'pm_name': 'Katarzyna Sroga@', 'pm_email': 'katarzyna.sroga@nhs.net'},
    'The-Chelsea-Practice': {'pm_name': 'Kas Shackleford', 'pm_email': 'k.shackleford@nhs.net'},
    'The-Good-Practice': {'pm_name': 'Cameron McIvor', 'pm_email': 'cameron.mcivor@nhs.net'},
}


# Sending actual email
params: resend.Emails.SendParams = {
    "from": "AI-MedReview AI Agent <hello@attribut.me>",
    "to": [surgery_info['Earls-Court-Surgery']['pm_email']],
    "subject": "AI-MedReview Alert - Negative Reviews",
    "html": f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>AI MedReview Alert</title>
</head>
<body style="margin:0; padding:0; background-color:#cbd5e1; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color:#1e293b;">

  <table align="center" border="0" cellpadding="0" cellspacing="0" width="600" style="background-color:white; border-radius:8px; overflow:hidden; box-shadow:0 4px 8px rgba(0,0,0,0.05); margin-top:40px;">

    <!-- Header -->
    <tr>
      <td bgcolor="#075985" align="center" style="padding: 20px 0;">
        <h2 style="color:white; margin:0; font-size:24px;">AI MedReview - Alert</h2>
      </td>
    </tr>

    <!-- Body Content -->
    <tr>
      <td style="padding: 30px 40px;">
        <p style="margin: 0 0 15px;">Dear {{ surgery_info['Earls-Court-Surgery']['pm_name'] }},</p>

        <p style="margin: 0 0 10px;">You have received the following negative review that requires your attention:</p>

        <div style="background-color:#e0f2fe; padding:15px; border-left:4px solid #075985; margin:20px 0;">
          <p style="margin:0;"><strong>Review 1:</strong> The surgery is awful.</p>
          <p style="margin:5px 0 0;"><strong>Sentiment Score:</strong> Negative (0.95)</p>
        </div>

        <p style="margin:20px 0 0;">Please review and take appropriate action.</p>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td align="center" bgcolor="#f1f5f9" style="padding: 15px; font-size: 12px; color: #64748b;">
        Regards, <br>
        AI MedReview Agent - Clive
      </td>
    </tr>

  </table>

</body>
</html>""",
}

r = resend.Emails.send(params)
print(r)
