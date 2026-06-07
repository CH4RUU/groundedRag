import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GOOGLE_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
r = requests.get(url)
if r.status_code == 200:
    models = r.json().get("models", [])
    print("Available models:")
    for m in models:
        methods = m.get("supportedGenerationMethods", [])
        name = m.get("name")
        print(f"- {name} (methods: {methods})")
else:
    print(f"Error {r.status_code}: {r.text}")
