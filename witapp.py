from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import time
import PyPDF2
import docx
from transformers import pipeline

# ===== Flask App Setup =====
app = Flask(__name__)

# Optional: store settings in memory
user_settings = None

# ===== File Upload Configuration =====
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== Text Extraction =====
def extract_text(file_path):
    ext = file_path.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == "pdf":
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif ext == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# ===== Hugging Face AI Setup =====
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    max_length=300
    # use_auth_token="YOUR_TOKEN"  # uncomment only if using hosted API
)

def generate_summary(text):
    prompt = f"Summarize the following content for students:\n\n{text}\n\nSummary:"
    output = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,       # higher = more random
    top_k=50,              # limits choices to top 50 tokens
    top_p=0.9,             # nucleus sampling
    repetition_penalty=1.2)
    return output[0]["generated_text"]

def generate_quiz(text):
    prompt = f"Create a short multiple-choice quiz based on the following content:\n\n{text}\n\nQuiz:"
    output = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,       # higher = more random
    top_k=50,              # limits choices to top 50 tokens
    top_p=0.9,             # nucleus sampling
    repetition_penalty=1.2) # discourages repetition)
    return output[0]["generated_text"]

def generate_explanation(text):
    prompt = f"Explain the following concepts in simple terms for students:\n\n{text}\n\nExplanation:"
    output = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,       # higher = more random
    top_k=50,              # limits choices to top 50 tokens
    top_p=0.9,             # nucleus sampling
    repetition_penalty=1.2)
    return output[0]["generated_text"]

# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def settings():
    global user_settings
    if request.method == "POST":
        # Save the settings from the form
        user_settings = {
            "game_type": request.form.get("game_type"),
            "difficulty": request.form.get("difficulty"),
            "timer": True if request.form.get("timer") == "on" else False,
            "bg_color": request.form.get("bg_color", "#ffffff"),
            "text_color": request.form.get("text_color", "#000000")
        }
        print("Saved settings:", user_settings)
    return render_template("settings.html", settings=user_settings)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "document" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["document"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        final_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], final_filename)
        file.save(file_path)

        # Extract text
        text = extract_text(file_path)

        # Generate AI outputs
        summary = generate_summary(text)
        quiz = generate_quiz(text)
        explanation = generate_explanation(text)

        # Render results in HTML template
        return render_template(
            "templateResults.html",
            filename=final_filename,
            summary=summary,
            quiz=quiz,
            explanation=explanation
        )
    return jsonify({"error": "File type not allowed"}), 400

# ===== Run Server =====
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)