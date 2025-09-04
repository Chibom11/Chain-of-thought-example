from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()

client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.getenv("API_KEY")  
)

SYSTEM_PROMPT = """
You're an expert AI Assistant in resolving user queries using chain of thought.
You work on START, PLAN and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be multiple steps.
Once you think enough PLAN has been done, finally you can give an OUTPUT.

Rules:
- Strictly Follow the given JSON output format
- Only run one step at a time.
- The sequence of steps is START (where user gives an input), PLAN (can be multiple times) and finally OUTPUT (which is going to the displayed to the user).

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT", "content": "string" }
"""

message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

user_query = input("👤 : ")
message_history.append({"role": "user", "content": user_query})

while True:
    response = client.chat.completions.create(
        model="Meta-Llama-3.3-70B-Instruct",
        response_format={"type": "json_object"},
        messages=message_history
    )

    raw_result = response.choices[0].message.content
    parsed_result = json.loads(raw_result)

    # Save assistant reply into history
    message_history.append({"role": "assistant", "content": raw_result})

    # Handle steps
    if parsed_result.get("step") == "START":
        print("🔥", parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "PLAN":
        print("🧠", parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "OUTPUT":
        print("🤖", parsed_result.get("content"))
        break
