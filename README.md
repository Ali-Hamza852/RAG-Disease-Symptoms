# 🏥 RAG-Disease-Symptoms: Context-Aware Medical Intelligence

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AliHamza852/RAG-Disease-Symptoms)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Generative AI](https://img.shields.io/badge/Focus-Generative%20AI-red)](https://github.com/Ali-Hamza852)

**RAG-Disease-Symptoms** is an advanced Retrieval-Augmented Generation pipeline designed to provide context-aware insights into medical symptoms and potential disease correlations. By bridging a curated medical knowledge base with Large Language Models (LLMs), this system ensures that responses are grounded in retrieved data rather than just parametric memory.

---

## 🚀 Live Demo
Interact with the diagnostic assistant here:
**[Medical RAG Assistant on Hugging Face](https://huggingface.co/spaces/AliHamza852/RAG-Disease-Symptoms)**

---

## ✨ Key Features
* **Grounded Responses:** Utilizes RAG to minimize hallucinations by fetching relevant medical context before generating answers.
* **Vector Search:** High-speed retrieval of disease-symptom correlations using semantic similarity.
* **Specialized Domain Focus:** Tailored for medical symptom prediction and health-related query resolution.
* **Modern Stack:** Implemented using leading Generative AI frameworks for robust document indexing and retrieval.

---

## 🛠️ Technical Architecture

### Retrieval Pipeline
* **Vector Store:** Optimized indexing for fast symptom-to-disease lookups.
* **Embedding Model:** Converts medical queries into dense vectors to capture semantic nuances.
* **RAG Strategy:** Dynamic context injection into the prompt to provide evidence-based insights.

### Tech Stack
* **Framework:** LangChain / LlamaIndex
* **UI:** Gradio 
* **Deployment:** Hugging Face Spaces (GPU/CPU Optimized)

---

## 📂 Project Structure
```text
├── app.py              # Main application logic & RAG pipeline
├── data/               # Medical knowledge base / Symptom datasets
├── vector_db/          # Pre-computed vector embeddings (FAISS/Chroma)
├── requirements.txt    # Environment dependencies
└── README.md           # Project Documentation

⚠️ Disclaimer

This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.
💻 Local Setup

    Clone the repository:
    Bash

git clone [https://github.com/Ali-Hamza852/RAG-Disease-Symptoms.git](https://github.com/Ali-Hamza852/RAG-Disease-Symptoms.git)
cd RAG-Disease-Symptoms

Install dependencies:
Bash

pip install -r requirements.txt

Configure API Keys:
Create a .env file and add your necessary LLM provider keys (e.g., OPENAI_API_KEY).

Launch:
Bash

    python app.py

🤝 Team & Acknowledgments

Developed by Ali Hamza
