# Company Policy Assistant — PS 34

A Retrieval-Augmented Generation (RAG) chatbot that answers employee questions about leave, travel and reimbursements using uploaded HR policy PDFs. It displays the source policy filename and page when available.

## Features
- Upload multiple HR policy PDFs.
- Extract PDF text, split it into chunks and index it in ChromaDB.
- Retrieve relevant chunks for each question.
- Generate grounded answers using Ollama or Gemini.
- Show source policy filenames and page numbers.
- Refuse to invent information when the policies do not contain an answer.

## Tech stack
Python, Streamlit, LangChain, ChromaDB, Sentence Transformers, PyPDF, Ollama (Qwen 2.5 3B) or Gemini API.

## Run locally (Mac / Windows)
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:
   - Mac: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. For local Ollama, install Ollama, then run:
   `ollama pull qwen2.5:3b`
5. Start the app:
   `streamlit run app.py`
6. In the sidebar select Ollama (local), keep model `qwen2.5:3b`, then click **Process policy documents**. The bundled sample PDFs are selected by default.

### Gemini option
Select Gemini API in the sidebar, enter your own Gemini API key, then process the PDFs and ask questions. Never commit API keys to GitHub.

## Sample questions
- How many casual leave days are available each year?
- What documents are required for a travel reimbursement claim?
- What is the deadline for submitting a local travel claim?
- Can unused casual leave be carried forward?
- Does the policy mention relocation reimbursement?

## Important
The included HR policy PDFs are fictional sample data created only to demonstrate the application. They are not actual company policies. Replace them with authorized policy PDFs for real use. If an answer is absent from the retrieved documents, the assistant is instructed to direct the employee to HR.

## Project structure
```
Company_Policy_Assistant_PS34/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── policies/
│   ├── Sample_Leave_Policy.pdf
│   ├── Sample_Travel_Policy.pdf
│   └── Sample_Reimbursement_Policy.pdf
└── screenshots/
```
