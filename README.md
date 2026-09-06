# Multi-Agent RAG System (LangGraph + Gemini)

A lightweight, two-node multi-agent RAG (Retrieval-Augmented Generation) pipeline built using **LangGraph** and powered by **Google Gemini 3.6 Flash**.

The system takes user queries, extracts relevant snippets from a local knowledge base using keyword filtering, and passes the retrieved data to a report generator agent to write a structured answer without hallucination.

Reference instruction: https://docs.google.com/document/d/1SX0s7Kqh75IPYSjdR7ziiCuroGthO3KO/edit?pli=1

Reference articles:
- https://theaiengineer.substack.com/p/what-is-semantic-search-f45

---

## 🛠️ Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.10+ installed.

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

Load all required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

```

---

## 🔑 Environment Setup

Before running the workflow, export your **Google Gemini API Key** as an environment variable.

### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="your_actual_gemini_api_key_here"

```

### Windows (Command Prompt)

```cmd
set GEMINI_API_KEY=your_actual_gemini_api_key_here

```

### macOS / Linux

```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"

```

---

## 🚀 Execution

Run the main pipeline:

```bash
python main.py

```

---

## 📸 Execution Results

Below is the verified output screenshot showing successful sequential data retrieval and answer synthesis using `gemini-3.6-flash`.

Question 01
<img width="1608" height="544" alt="Screenshot 2026-09-03 095424" src="https://github.com/user-attachments/assets/d4caa073-5abc-4503-93ba-b12348c99bf4" />

Question 02
<img width="1616" height="748" alt="Screenshot 2026-09-03 095211" src="https://github.com/user-attachments/assets/7bd2e8ad-6791-49e7-878a-8912478060ff" />

Question 03
<img width="1624" height="674" alt="Screenshot 2026-09-03 094649" src="https://github.com/user-attachments/assets/7c58aa3a-5e86-4d75-aa4e-db22c89b28c0" />

Question 04
<img width="1610" height="630" alt="Screenshot 2026-09-03 100114" src="https://github.com/user-attachments/assets/cbe55211-7200-47df-b7f6-3b84d887b5f6" />

Semantic test 01
<img width="1590" height="744" alt="image" src="https://github.com/user-attachments/assets/62499011-0b3f-4584-a38b-34c7434f87f8" />



