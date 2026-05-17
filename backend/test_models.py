import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print("Checking available models...")
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("\n--- AVAILABLE GEMINI MODELS ---")
        for m in data.get('models', []):
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(f"- {m['name']}")
        print("-------------------------------\n")
except Exception as e:
    print(f"Error fetching models: {e}")
