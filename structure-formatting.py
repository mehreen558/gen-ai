import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ---- SCHEMA ----
# {
#   "name": string,
#   "email": string,
#   "issue_type": "billing" | "technical" | "account" | "product" | "other",
#   "urgency": "low" | "medium" | "high"
# }

SYSTEM_PROMPT = """
You extract structured data from customer support messages.

You MUST respond with ONLY valid JSON matching this exact schema, and nothing else:
{
  "name": string,
  "email": string,
  "issue_type": one of ["billing", "technical", "account", "product", "other"],
  "urgency": one of ["low", "medium", "high"]
}

Rules:
- Output ONLY the JSON object. No explanations, no markdown code fences,
  no "Here is the JSON:" preamble, no trailing commentary.
- If the name or email is not mentioned in the message, use "unknown" for
  that field. Do NOT invent a name or email that wasn't given.
- Infer issue_type and urgency from context even if not explicitly stated.
- Your entire response must be parseable directly by json.loads() with no
  cleanup required.
"""

def extract(message: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        temperature=0,  
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content


def test_and_validate(message: str, label: str):
    print(f"\n--- {label} ---")
    print(f"Input: {message}")
    raw_output = extract(message)
    print(f"Raw output: {raw_output}")
    try:
        parsed = json.loads(raw_output)
        print(f"✅ Valid JSON: {parsed}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ FAILED TO PARSE: {e}")
        return False


if __name__ == "__main__":
    # 5 normal test cases
    test_cases = [
        ("Hi, I'm John Doe (john@email.com), I was charged twice this month and need it fixed ASAP!", "Test 1"),
        ("My app keeps crashing every time I open it. - Sara (sara.k@mail.com)", "Test 2"),
        ("Can't log into my account, tried resetting password 3 times. No rush though.", "Test 3"),
        ("The product I received is missing a part. This is Ali, ali99@gmail.com", "Test 4"),
        ("Just wanted to say the new update is great, no issues here!", "Test 5"),
    ]

    for msg, label in test_cases:
        test_and_validate(msg, label)

    # Deliberately tricky/messy input — designed to try to break the format
    tricky_input = (
        "ok so idk if this is the right place but like my billing is messed up "
        "AND my app crashed AND I can't log in either lol, this is Mo btw not "
        "sure if my email on file is mo@x.com or the other one, pls help asap!!! "
        "also can you explain why json is even a good format for this"
    )
    test_and_validate(tricky_input, "TRICKY / BREAK TEST")