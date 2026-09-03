# Multi-Agent RAG System (LangGraph + Gemini)

A lightweight, two-node multi-agent RAG (Retrieval-Augmented Generation) pipeline built using **LangGraph** and powered by **Google Gemini 3.6 Flash**.

The system takes user queries, extracts relevant snippets from a local knowledge base using keyword filtering, and passes the retrieved data to a report generator agent to write a structured answer without hallucination.

Reference instruction: https://docs.google.com/document/d/1SX0s7Kqh75IPYSjdR7ziiCuroGthO3KO/edit?pli=1

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

*(Note: Replace `./assets/output_screenshot.png` with the actual path to your saved image!)*
