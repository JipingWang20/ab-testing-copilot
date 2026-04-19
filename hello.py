import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
)

print(response.content[0].text)
