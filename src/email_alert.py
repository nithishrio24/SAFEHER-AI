import smtplib, os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SENDER   = os.getenv("ALERT_EMAIL", "").strip()
PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()
TO_LIST  = os.getenv("ALERT_TO_EMAILS", "").strip()

def send_alert_email(transcript, confidence,
                     latitude=None, longitude=None):

    print(f"[EMAIL] FROM={SENDER}")
    print(f"[EMAIL] TO={TO_LIST}")
    print(f"[EMAIL] PASS={'SET' if PASSWORD else 'MISSING'}")

    if not SENDER or not PASSWORD or not TO_LIST:
        print("[EMAIL] ❌ .env values missing")
        return False

    recipients = [e.strip() for e in TO_LIST.split(",")]
    pct = round(confidence * 100)
    now = datetime.now().strftime("%d %b %Y %I:%M %p")

    if latitude and longitude:
        map_url = f"https://maps.google.com/?q={latitude},{longitude}"
        loc_html = f'<p>📍 <a href="{map_url}">Open Location</a></p>'
        loc_txt  = f"📍 Location: {map_url}"
    else:
        loc_html = "<p>📍 Location: Not available</p>"
        loc_txt  = "📍 Location: Not available"

    plain = f"""
🆘 SAFEHER AI EMERGENCY ALERT
Time: {now}
Speech: "{transcript}"
Confidence: {pct}%
{loc_txt}
      """.strip()

    html = f"""
      <div style="font-family:'Times New Roman',serif;
                  max-width:520px;margin:auto;
                  border:3px solid #ff2d55;border-radius:12px;
                  overflow:hidden;">
        <div style="background:#ff2d55;padding:28px;text-align:center;">
          <h1 style="color:white;margin:0;font-size:32px;
                     font-weight:bold;font-family:'Times New Roman',serif;">
            🆘 EMERGENCY ALERT
          </h1>
          <p style="color:white;margin:8px 0 0;font-size:16px;">
            SafeHer AI detected distress
          </p>
        </div>
        <div style="padding:28px;background:#fff;">
          <p style="font-size:16px;">⏰ <b>Time:</b> {now}</p>
          <p style="font-size:16px;">🎤 <b>Detected:</b>
            <span style="color:#ff2d55;font-weight:bold;">
              "{transcript}"
            </span>
          </p>
          <p style="font-size:16px;">
            📊 <b>Confidence:</b> {pct}%
          </p>
          {loc_html}
          <hr style="border:none;border-top:1px solid #eee;">
          <p style="color:#999;font-size:12px;">
            Sent by SafeHer AI. If this was a mistake, ignore this email.
          </p>
        </div>
      </div>
      """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🆘 SafeHer ALERT — {pct}% Distress Detected"
        msg["From"]    = SENDER
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(SENDER, PASSWORD)
            s.sendmail(SENDER, recipients, msg.as_string())

        print(f"[EMAIL] ✅ Sent to {recipients}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[EMAIL] ❌ Auth failed — wrong app password")
        return False
    except Exception as e:
        print(f"[EMAIL] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        return False
