import os
import re
import pdfplumber
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

PDF_PATH = "research-paper.pdf"
GROQ_MODEL = "llama-3.1-8b-instant"
HEADER_RE = re.compile(r"^\s*The EUROCALL Review,\s*Volume\s*25,\s*No\.\s*2,\s*September\s*2017\s*$")
PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


def strip_header_footer(page_text, lookahead_lines=5):
    """
    Remove the repeating running header ('The EUROCALL Review, Volume 25,
    No. 2, September 2017') and the page-number footer from a page's raw
    extracted text.

    pypdf preserves the PDF's internal content-stream order, not visual
    top-to-bottom order, so on this document both the header line and the
    footer page number actually show up together near the TOP of each
    page's extracted text (confirmed by inspection), not at the very end.
    Only digit-only lines within the first few lines are treated as the
    footer, so a genuine standalone number later in the body (e.g. inside
    a list or citation) is never touched.
    """
    lines = page_text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            continue
        if i < lookahead_lines and PAGE_NUM_RE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append(strip_header_footer(raw))
    return pages


def extract_real_tables(pdf_path, min_cols=3, min_rows=3):
    """
    pdfplumber flags almost every text block as a 1-column 'table', so we
    filter for genuine multi-column tables only. Returns a list of
    {page, header, rows} dicts, one per detected table.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for raw_table in page.extract_tables():
                rows = [r for r in raw_table if any(c not in (None, "") for c in r)]
                if len(rows) < min_rows:
                    continue
                n_cols = max(len(r) for r in rows)
                if n_cols < min_cols:
                    continue
                tables.append({"page": page_num, "rows": rows})
    return tables


def table_to_markdown(rows, header):
    """Render a list of {col: value} row-dicts as a Markdown table string.
    (Utility kept for optional debugging/printing; not used in the main
    per-row chunking pipeline below.)"""
    cols = list(header)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def build_table1_chunks(pdf_path):
    """
    Table 1 ('The students' mobile devices usage descriptions') is hand-parsed
    here because it spans two PDF pages and has a merged/rotated header that
    generic table detection can't cleanly label.

    Produces ONE CHUNK PER STUDENT ROW (not per page/table). Cramming all 7-13
    rows of a page into a single chunk dilutes a specific student's signal in
    TF-IDF (shared words like "female"/"smartphone" across many rows drown out
    the one row that actually answers a targeted query like "what did S14
    use?"). A per-row chunk keeps each student's data as its own precise,
    self-contained, retrievable unit — while still spelling out every column
    label so meaning isn't lost outside table structure.
    """
    raw_tables = extract_real_tables(pdf_path)

    chunks = []
    current_level = ""
    for t in raw_tables:
        data_rows = [r for r in t["rows"] if r[1] and re.match(r"^S\d+$", str(r[1]).strip())]
        for r in data_rows:
            level_cell = (r[0] or "").replace("\n", " ").strip()
            if level_cell:
                current_level = level_cell
            row_text = (
                f"Table 1 row - Student {r[1].strip()}: "
                f"Sex = {(r[2] or '').strip()}; "
                f"Device used = {(r[3] or '').strip()}; "
                f"Years used mobile devices for English = {(r[4] or '').strip()}; "
                f"Self-assessed experience = {(r[5] or '').strip()}. "
                f"(Year/level of study: {current_level})"
            )
            chunks.append({"page": t["page"], "text": row_text})

    if chunks:
        summary_lines = [c["text"] for c in chunks]
        chunks.append({
            "page": chunks[0]["page"],
            "text": "Table 1 full summary (all 20 students):\n" + "\n".join(summary_lines)
        })
    return chunks


def chunk_text(pages, chunk_size=80, overlap=60):
    """Chunk by words, with overlap, keeping track of source page."""
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split(" ")
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append({
                    "page": page_num,
                    "text": " ".join(chunk_words)
                })
            i += chunk_size - overlap
    return chunks


def build_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, chunks, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def retrieve_hybrid(query, prose_vectorizer, prose_matrix, prose_chunks,
                     table_vectorizer, table_matrix, table_chunks,
                     top_k_prose=7, top_k_table=2):
    """
    Retrieve from prose and table chunks in SEPARATE vector spaces, then
    merge. Mixing them in one shared TF-IDF space lets short, repetitive
    table rows (e.g. 'Sex = female; Device used = smartphone...') win on
    cosine similarity purely because they're short and densely match a few
    query words — even when a longer prose chunk actually contains the
    better answer. Retrieving each source type independently guarantees
    prose chunks always get a fair shot regardless of table row count.
    """
    prose_results = retrieve(query, prose_vectorizer, prose_matrix, prose_chunks, top_k=top_k_prose)
    table_results = retrieve(query, table_vectorizer, table_matrix, table_chunks, top_k=top_k_table)
    combined = prose_results + table_results
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


SYSTEM_PROMPT = (
    "You are a RAG assistant. Answer the user's question using ONLY the "
    "provided context chunks from a research paper. If the answer is not "
    "present in the context, say clearly 'Not found in the provided context' "
    "instead of guessing. Do not use outside knowledge. Cite the page "
    "number(s) you used."
)


def generate_answer(client, question, retrieved):
    context = "\n\n".join(
        f"[Page {c['page']}, similarity={score:.3f}]\n{c['text']}"
        for c, score in retrieved
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    pages = extract_text(PDF_PATH)
    prose_chunks = chunk_text(pages)
    table_chunks = build_table1_chunks(PDF_PATH)
    prose_vectorizer, prose_matrix = build_index(prose_chunks)
    table_vectorizer, table_matrix = build_index(table_chunks)
    print(f"Extracted {len(pages)} pages -> {len(prose_chunks)} prose chunks "
          f"+ {len(table_chunks)} table chunks (indexed separately)\n")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    questions = [
        "How many students participated in the study and how were they split by year/level?",
        "What was the average age of the participants?",
        "What data collection method was used, and in what language were interviews conducted?",
        "What percentage of interviewees felt mobile devices helped them study more effectively?",
        "According to the paper, did teachers explicitly recommend or require the use of mobile devices in class?",
        "Which device did student S14 use, and how experienced were they?",  
        "What was the sample size in a similar prior study by Byrne & Diem (2014)?",  # not in doc
    ]

    for q in questions:
        print("=" * 90)
        print("Q:", q)
        results = retrieve_hybrid(q, prose_vectorizer, prose_matrix, prose_chunks,table_vectorizer, table_matrix, table_chunks)
        for chunk, score in results:
            print(f"  [retrieved page {chunk['page']}, sim={score:.3f}]")
        answer = generate_answer(client, q, results)
        print("\nA:", answer)
        print()