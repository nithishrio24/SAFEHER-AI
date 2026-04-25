import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

FAST2SMS_KEY = os.getenv("FAST2SMS_KEY")

def send_distress_sms(phone_number, user_name, 
                      transcript, confidence, 
                      latitude, longitude):
    
    maps_link = f"https://maps.google.com/?q={latitude},{longitude}"
    now = datetime.now().strftime('%d %b %Y, %I:%M %p')
    
    message = (
        f"SAFEHER AI EMERGENCY ALERT "
        f"{user_name} may be in danger! "
        f"Heard: {transcript} "
        f"Confidence: {confidence}% "
        f"Time: {now} "
        f"Location: {maps_link} "
        f"Call emergency: 112"
    )
    
    # Remove +91 if present, keep only 10 digits
    phone = str(phone_number).replace('+91','').replace(' ','').strip()
    
    url = "https://www.fast2sms.com/dev/bulkV2"
    
    payload = {
        "authorization": FAST2SMS_KEY,
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": phone
    }
    
    headers = {
        "cache-control": "no-cache"
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers)
        result = response.json()
        
        if result.get("return") == True:
            print(f"✅ SMS sent to {phone}")
            return {"status": "sent", "phone": phone}
        else:
            print(f"❌ SMS failed: {result}")
            return {"status": "failed", "phone": phone, "error": str(result)}
            
    except Exception as e:
        print(f"❌ SMS error: {str(e)}")
        return {"status": "error", "phone": phone, "error": str(e)}


def send_all_contacts(contacts_list, user_name,
                      transcript, confidence,
                      latitude, longitude):
    results = []
    for contact in contacts_list:
        result = send_distress_sms(
            contact['phone'],
            user_name,
            transcript,
            confidence,
            latitude,
            longitude
        )
        result['name'] = contact['name']
        results.append(result)
    return results


def send_cancel_sms(phone_number, user_name):
    phone = str(phone_number).replace('+91','').replace(' ','').strip()
    
    message = (
        f"SafeHer AI Update: "
        f"{user_name} is SAFE now. "
        f"Previous alert has been cancelled. "
        f"No action needed."
    )
    
    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "authorization": FAST2SMS_KEY,
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": phone
    }
    
    try:
        response = requests.post(url, data=payload)
        result = response.json()
        if result.get("return") == True:
            print(f"✅ Cancel SMS sent to {phone}")
        else:
            print(f"❌ Cancel SMS failed: {result}")
    except Exception as e:
        print(f"❌ Cancel SMS error: {str(e)}")
