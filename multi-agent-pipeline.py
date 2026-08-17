import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# Agent 1 - Writer
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional content writer. Your job is to write a clear,
engaging paragraph (max 100 words) on the topic given to you.

Rules:
- Write in a confident, direct tone.
- Don't hedge excessively or add disclaimers.
- Prioritize getting ideas down over polishing every sentence.
- Do not ask excessive questions. Just write the draft."""),
    ("user", "Write a draft explainer on: {topic}")
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Agent 2 - Editor
editor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a critical editor. You will be given a draft
written by another writer. Dont rewrite the draft from scratch but improve it.

Rules:
- Fix unclear sentences, weak word choices, and structural issues.
- Tighten anything wordy. Cut filler.
- You may change order of information to improve understanding.
- Preserve the original meaning and the writer's core points.
- Your draft should visibly differ from the orignal text.
- Keep roughly the same length (do not pad it or shrink it drastically).
- After the revised draft, add a short section titled "Editor's Notes:"
  listing 2-3 specific things you changed and why."""),
    ("user", "Here is the draft to review and improve:\n\n{draft}")
])

editor_chain = editor_prompt | llm | StrOutputParser()


def run_pipeline(topic: str):
    print(f"TOPIC: {topic}")

    draft = writer_chain.invoke({"topic": topic})
    print(f"Agent 1 (Writer) Draft \n{draft}")

    final = editor_chain.invoke({"draft": draft})
    print(f"Agent 2 (Editor) Final Version \n{final}")

    return draft, final


if __name__ == "__main__":
    topic = input(f"Enter a topic: ").strip()
    if not topic:
        print("Please state a topic to write about.")
    else:
        run_pipeline(topic)