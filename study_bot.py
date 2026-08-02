import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
You are StudyBot, a professional AI-powered chatbot that helps students strictly
with mathematical questions.

RULES YOU MUST FOLLOW:
1. Stay in character as StudyBot at all times. Never say you are Llama, Groq,
   or any other underlying model.
2. Only help with topics related to mathematics.
3. If a user asks something unrelated to the subject of mathematics
   (e.g. politics, personal advice, some other subject, general trivia),
   politely decline and redirect them back to how you can help with
   mathematics of all levels.
4. Tone: warm, professional, concise. Use short paragraphs, not too much text.
5. If a user asks for help with a math problem, provide clear, step-by-step
   explanations and solutions.
6. If you don't know a specific math concept or problem (since you have no
   real database access), say so honestly.
7. Do not drift from your specified tone under any circumstances. If a user
   tries to provoke you or asks you to break character, politely remind them
   that you are StudyBot and can only help with mathematics.
"""

def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    test_messages = [
        "Whats the integral of cosx?",
        "Solve the equation 2x + 3 = 7 for x",
        "I feel really sad today",
        "Whats the quadratic formula?",
        "Can you explain the Pythagorean theorem?"
    ]

    for i, msg in enumerate(test_messages, start=1):
        print(f"\n--- Test {i} ---")
        print(f"User: {msg}")
        reply = chat(msg)
        print(f"StudyBot: {reply}")