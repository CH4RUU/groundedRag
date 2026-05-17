import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
url_template = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key=" + api_key

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-pro-latest"
]

payload = json.dumps({
    "contents": [{"parts": [{"text": "Hello"}]}]
}).encode('utf-8')

print("Testing Gemini model quotas...\n")
for model in models_to_test:
    url = url_template.format(model)
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print(f"SUCCESS: {model}")
    except urllib.error.HTTPError as e:
        err_msg = "Unknown error"
        try:
            error_details = json.loads(e.read().decode())
            err_msg = error_details.get("error", {}).get("message", err_msg)
        except:
            pass
        print(f"FAILED:  {model} -> {e.code} {err_msg}")
    except Exception as e:
        print(f"ERROR:   {model} -> {e}")
