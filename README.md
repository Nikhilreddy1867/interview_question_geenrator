# Interview Question Generator

An AI-powered application that generates technical interview questions based on skills extracted from resumes.

## Features

- PDF resume parsing
- Skill extraction using custom SpaCy model
- Technical interview question generation using T5 model
- RESTful API endpoints
- CORS support for frontend integration

## Tech Stack

- Backend: Flask (Python)
- Machine Learning: 
  - T5 for question generation
  - SpaCy for skill extraction
- PDF Processing: PyMuPDF
- Frontend Integration: CORS enabled

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Nikhilreddy1867/interview_question_geenrator.git
cd interview_question_geenrator
```

2. Install dependencies:
```bash
cd flask_backend
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The server will start on `http://localhost:9000`

## API Endpoints

### POST /api/generate-questions
Generates interview questions based on skills extracted from a PDF resume.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - pdf: PDF file
- Headers:
  - Authorization: Bearer {token}

**Response:**
```json
{
    "skills": ["extracted skills"],
    "questions": {
        "skill": ["generated questions"]
    },
    "questionId": "unique_id"
}
```

## Environment Variables

- `PORT`: Server port (default: 9000)
- `FRONTEND_URL`: Frontend application URL
- `EXPRESS_URL`: Express backend URL

## License

MIT License 