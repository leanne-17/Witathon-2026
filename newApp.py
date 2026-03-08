from flask import Flask, render_template, request, jsonify, session
import os
import json
import re
import time
import PyPDF2
import docx
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ===== Flask App Setup =====
app = Flask(__name__)
app.secret_key = "wit-study-secret-key-2024"

# ===== File Upload Configuration =====
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

# Flan-T5 has a 512-token encoder limit. ~1800 chars fits safely.
# For per-item calls we use a shorter chunk so the specific sentence
# context stays tight and relevant.
MAX_TEXT_CHARS = 1800
CHUNK_CHARS    = 900   # used when calling the model once per item

# ===== AI Model Setup =====
# google/flan-t5-base is a seq2seq model fine-tuned to follow instructions.
# It actually reads the input text and answers the task — unlike gpt-neo.
# Upgrade path: swap to "google/flan-t5-large" for better quality (needs ~1 GB RAM extra).
print("Loading Flan-T5 model... (first run downloads ~900 MB, cached after that)")
_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
_model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
_model.eval()
print("Model ready.")

# ===== Helpers =====

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path):
    """Extract plain text from txt, pdf, or docx."""
    ext = file_path.rsplit(".", 1)[1].lower()
    text = ""
    try:
        if ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == "pdf":
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif ext == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"Text extraction error: {e}")
    return text.strip()


def truncate(text, max_chars=MAX_TEXT_CHARS):
    """Hard-truncate to stay within Flan-T5's encoder token budget."""
    return text[:max_chars] if len(text) > max_chars else text


def run(prompt, max_new_tokens=120):
    """
    Call Flan-T5 directly via tokenizer + model.generate().
    Bypasses pipeline entirely — pipeline doesn't support T5ForConditionalGeneration
    under 'text-generation' in newer transformers versions.
    """
    try:
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,   # Flan-T5 encoder hard limit
        )
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        result = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"\n--- PROMPT ---\n{prompt[:200]}\n--- OUTPUT ---\n{result}\n")
        return result
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


def split_sentences(text, n=5):
    """
    Split the notes into roughly n equal-ish sentence groups so each
    per-item call gets a focused, relevant chunk of the notes.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) > 10]
    if not sentences:
        # Fall back to word-based chunking
        words = text.split()
        chunk_size = max(1, len(words) // n)
        sentences = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    # Distribute into n groups
    group_size = max(1, len(sentences) // n)
    groups = []
    for i in range(n):
        chunk = " ".join(sentences[i * group_size: (i + 1) * group_size])
        if chunk:
            groups.append(chunk[:CHUNK_CHARS])
    # If fewer groups than n, repeat the last one
    while len(groups) < n:
        groups.append(groups[-1] if groups else text[:CHUNK_CHARS])
    return groups[:n]


# ===== AI Generation Functions =====

def generate_summary(text):
    """
    Summarise the full notes in plain text.
    Flan-T5 handles 'summarize:' as a direct task prefix.
    """
    prompt = f"Summarize the following study notes in 4 clear sentences:\n\n{truncate(text)}"
    result = run(prompt, max_new_tokens=160)
    return result if result else "Could not generate a summary. Please try again."


def generate_quiz(text, n=4):
    """
    Generate n quiz questions by asking the model one targeted question per
    sentence-chunk. This ensures each question is grounded in actual content.

    Returns: [{"question": str, "options": [str,str,str,str], "answer": int}, ...]
    """
    chunks  = split_sentences(text, n)
    questions = []

    for chunk in chunks:
        # Step 1: generate a question from this chunk
        q_prompt = (
            f"Based on this text, write one quiz question that tests understanding:\n{chunk}"
        )
        question = run(q_prompt, max_new_tokens=60)
        if not question or len(question) < 8:
            continue

        # Step 2: get the correct answer
        a_prompt = f"Answer this question using the text below.\nQuestion: {question}\nText: {chunk}"
        correct = run(a_prompt, max_new_tokens=40)
        if not correct:
            correct = "See your notes"

        # Step 3: generate 3 plausible wrong answers
        wrong_prompt = (
            f"Give 3 short, plausible but incorrect answers for this question. "
            f"Separate them with '|'.\nQuestion: {question}\nCorrect answer: {correct}"
        )
        wrong_raw = run(wrong_prompt, max_new_tokens=60)
        wrongs = [w.strip() for w in wrong_raw.split("|") if w.strip() and w.strip() != correct]
        # Fallback distractors if the model gives us too few
        fallbacks = ["None of the above", "All of the above", "Cannot be determined"]
        while len(wrongs) < 3:
            wrongs.append(fallbacks[len(wrongs) % len(fallbacks)])
        wrongs = wrongs[:3]

        # Shuffle correct answer into a random position
        import random
        correct_idx = random.randint(0, 3)
        options = wrongs[:correct_idx] + [correct] + wrongs[correct_idx:]

        questions.append({
            "question": question,
            "options":  options[:4],
            "answer":   correct_idx,
        })

    if not questions:
        questions = [{
            "question": "What is the main topic of these notes?",
            "options":  ["The key idea presented", "An unrelated concept", "A supporting detail", "None of the above"],
            "answer":   0,
        }]
    return questions


def generate_flashcards(text, n=5):
    """
    Generate n flashcards — one per sentence chunk.
    Each card: {"term": str, "definition": str}
    """
    chunks = split_sentences(text, n)
    cards  = []

    for chunk in chunks:
        # Extract the key term
        term_prompt = (
            f"What is the most important term or concept in this text? "
            f"Reply with only the term (a few words):\n{chunk}"
        )
        term = run(term_prompt, max_new_tokens=20)
        if not term or len(term) < 2:
            continue

        # Define it
        def_prompt = (
            f"Write a short definition of '{term}' based on this text "
            f"(one sentence only):\n{chunk}"
        )
        definition = run(def_prompt, max_new_tokens=60)
        if not definition:
            definition = chunk[:120]

        cards.append({"term": term, "definition": definition})

    if not cards:
        cards = [{"term": "Key Concept", "definition": "Review your notes to identify the main ideas."}]
    return cards


def generate_matching(text, n=5):
    """
    Generate n matching pairs — one per sentence chunk.
    Each pair: {"left": term, "right": definition}
    """
    chunks = split_sentences(text, n)
    pairs  = []

    for chunk in chunks:
        term_prompt = (
            f"What is the key term or concept in this text? "
            f"Reply with only the term (a few words):\n{chunk}"
        )
        term = run(term_prompt, max_new_tokens=20)
        if not term or len(term) < 2:
            continue

        desc_prompt = (
            f"Describe '{term}' in one short sentence based on this text:\n{chunk}"
        )
        description = run(desc_prompt, max_new_tokens=50)
        if not description:
            description = chunk[:100]

        pairs.append({"left": term, "right": description})

    if not pairs:
        pairs = [
            {"left": "Concept A", "right": "The first key idea from your notes"},
            {"left": "Concept B", "right": "The second key idea from your notes"},
        ]
    return pairs


def generate_boss_battle(text, n=4):
    """
    Same as quiz but with boss_hp_damage added per question.
    4 questions × 25 damage = 100 HP to defeat the boss.
    """
    questions = generate_quiz(text, n=n)
    for q in questions:
        q["boss_hp_damage"] = 25
    return questions


# ===== Routes =====

@app.route("/")
def index():
    """Landing / file upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Accepts a file + chosen mode, extracts text, generates AI content,
    then redirects to the appropriate result page.
    """
    if "document" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["document"]
    mode = request.form.get("mode", "summary")  # quiz | flashcards | summary | matching | boss

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use PDF, TXT or DOCX."}), 400

    # Save file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    final_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], final_filename)
    file.save(file_path)

    # Extract text
    text = extract_text(file_path)
    if not text:
        return jsonify({"error": "Could not extract text from file."}), 400

    # Route to correct generator and template
    if mode == "summary":
        data = generate_summary(text)
        return render_template("summary.html", summary=data)

    elif mode == "quiz":
        data = generate_quiz(text)
        return render_template("quiz.html", questions=json.dumps(data))

    elif mode == "flashcards":
        data = generate_flashcards(text)
        return render_template("flashcards.html", cards=json.dumps(data))

    elif mode == "matching":
        data = generate_matching(text)
        return render_template("matching.html", pairs=json.dumps(data))

    elif mode == "boss":
        data = generate_boss_battle(text)
        return render_template("boss_battle.html", questions=json.dumps(data))

    else:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

@app.route("/settings")
def settings():
    return render_template("settings.html", active_page="settings")
# ===== Run =====
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)