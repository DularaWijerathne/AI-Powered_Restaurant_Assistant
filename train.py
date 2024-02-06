import json

with open('Intents/intents.json') as f:
    data = json.load(f)

print(data['intents'])