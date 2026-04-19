import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()


# ============ 1. Define a fake tool ============
def add_numbers(a: float, b: float) -> float:
    return a + b


TOOLS = [
    {
        "name": "add_numbers",
        "description": "Add two numbers together and return the sum.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    }
]


# ============ 2. Ask a question that needs the tool ============
messages = [{"role": "user", "content": "What is 37 plus 58?"}]

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=TOOLS,
    messages=messages,
)

print("=" * 50)
print("FIRST RESPONSE")
print("=" * 50)
print("Stop reason:", response.stop_reason)
print("Content blocks:")
for block in response.content:
    print(f"  - type={block.type}")
    if block.type == "text":
        print(f"    text: {block.text}")
    elif block.type == "tool_use":
        print(f"    tool: {block.name}")
        print(f"    input: {block.input}")


# ============ 3. Execute the tool ============
tool_use = next(b for b in response.content if b.type == "tool_use")
result = add_numbers(**tool_use.input)

print("\n" + "=" * 50)
print(f"EXECUTED: {tool_use.name}({tool_use.input}) = {result}")
print("=" * 50)


# ============ 4. Send the result back, get the final answer ============
messages.append({"role": "assistant", "content": response.content})
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": str(result),
    }],
})

final = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=TOOLS,
    messages=messages,
)

print("\n" + "=" * 50)
print("FINAL ANSWER")
print("=" * 50)
print(final.content[0].text)
