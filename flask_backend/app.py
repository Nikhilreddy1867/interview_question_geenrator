from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import fitz
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
import os
from pathlib import Path

app = Flask(__name__)
# Allow all origins in development, configure specifically in production
CORS(app)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

# Use environment variables with defaults for configuration
PORT = int(os.environ.get('PORT', 9000))
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:5173')
EXPRESS_URL = os.environ.get('EXPRESS_URL', 'http://localhost:5000')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update model paths to be relative to the application
skill_extractor_model_path = os.path.join(BASE_DIR, "skill_extractor_model")
nlp = spacy.load(skill_extractor_model_path)

skills_list = [
    "Data Structures", "Languages and Frameworks", "Database and SQL",
    "Web Development", "Machine Learning", "Deep Learning", "Data Analysis",
    "Data Visualization", "Data Science", "C Programming", "Programming Languages",
    "CFrameworks", "Problem-Solving", "Communication", "Teamwork",
    "Automation System", "Backend Developer",
]

def extract_skills(text):
    try:
        doc = nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        filtered_skills = [skill for skill in skills if skill in skills_list]
        return list(set(filtered_skills))
    except Exception as e:
        raise

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.Document(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        raise

# Update the model paths
tokenizer = T5Tokenizer.from_pretrained(os.path.join(BASE_DIR, "qg_model_tokenizer"))
model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, "qg_model_ml"))
model.to(device)

def generate_questions(skills):
    try:
        model.eval()
        questions = {}
        with torch.no_grad():
            for skill in skills:
                input_text = f"Generate a technical interview question about {skill}:"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length").to(device)
                # Generate 3-4 questions per skill
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    num_beams=5,  # Increased for better diversity
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    temperature=0.9,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    num_return_sequences=3  # Generate 4 questions per skill
                )
                # Decode and store the generated questions
                skill_questions = []
                for output in outputs:
                    question = tokenizer.decode(output, skip_special_tokens=True).strip()
                    if question and question not in skill_questions:  # Avoid duplicates
                        skill_questions.append(question)
                # If fewer than 3 questions, keep all; otherwise, take up to 4
                questions[skill] = skill_questions[:4] if skill_questions else ["Could not generate question"]
        return questions
    except Exception as e:
        raise

@app.route('/api/generate-questions', methods=['POST'])
def generate_interview_questions_api():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authorization token missing'}), 401
    token = auth_header.split(' ')[1]

    try:
        text = extract_text_from_pdf(file)
        extracted_skills = extract_skills(text)
        questions = generate_questions(extracted_skills)

        express_url = f'{EXPRESS_URL}/api/questions/store'
        payload = {
            'filename': file.filename,
            'questions': questions,
        }
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        }
        response = requests.post(express_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to store questions'}), 500
        
        response_data = response.json()
        question_id = response_data.get('questionId')

        return jsonify({
            'skills': extracted_skills,
            'questions': questions,
            'questionId': question_id,
        })
    except Exception as e:
        print(f"Error in generate-questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Temporary debug route for root POST requests
@app.route('/', methods=['POST'])
def root_post():
    print(f"Received POST to root: Headers={request.headers}, Form={request.form}, Files={request.files}")
    return jsonify({'error': 'Invalid endpoint. Use /api/generate-questions'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)