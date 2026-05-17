import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('COHERE_API_KEY')

response = requests.get(
    "https://api.cohere.com/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    data = response.json()
    for model in data.get('models', []):
        if 'command' in model.get('name', ''):
            print(f"- {model['name']} (Endpoints: {', '.join(model.get('endpoints', []))})")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
