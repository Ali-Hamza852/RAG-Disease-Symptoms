import gradio as gr
import faiss
import json
import torch
import os
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. SETUP ---
print("🔄 System Starting...")
device = "cpu" # HF Spaces free tier uses CPU

# Load Embedding Model (Retrieval)
print("🧠 Loading Embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load Data Artifacts
print("📂 Loading Knowledge Base...")
documents = []
sources = []
index = None

try:
    # Load FAISS Index
    if os.path.exists("index.faiss"):
        index = faiss.read_index("index.faiss")
    
    # Load Documents
    if os.path.exists("docs.json"):
        with open("docs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            documents = data["documents"]
            sources = data["sources"]
            
    if index and documents:
        print(f"✅ Successfully loaded {len(documents)} documents.")
    else:
        print("❌ Error: Files missing. Did you upload index.faiss and docs.json?")
except Exception as e:
    print(f"❌ CRITICAL ERROR loading files: {e}")

# Load LLM (Generative Model)
# We use FLAN-T5-LARGE. It is smarter than 'base' and fits in the free tier memory.
print("🤖 Loading Instruction Model (Flan-T5-Large)...")
model_name = "google/flan-t5-large" 

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. LOGIC ---
def clean_text(text):
    """Helper to remove excessive JSON brackets for cleaner reading"""
    # Remove curly braces and quotes to make it look like normal text
    text = text.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_response(user_query):
    if index is None or not documents:
        return "Error: System not ready. Please check if files were uploaded correctly.", ""

    try:
        # A. Search (Retrieval)
        query_vector = embedder.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = index.search(query_vector, 3)
        
        # B. Retrieve Text
        retrieved_content = []
        retrieved_names = []
        
        for idx in indices[0]:
            if 0 <= idx < len(documents):
                raw_text = documents[idx]
                # Clean the JSON structure so the LLM doesn't get confused
                cleaned = clean_text(raw_text)
                retrieved_content.append(cleaned)
                retrieved_names.append(sources[idx])

        # C. Generate Answer (Generation)
        context_block = "\n -- \n".join(retrieved_content)
        
        # Strict Prompt for T5
        prompt = (
            f"Read the medical context below and answer the question truthfully.\n\n"
            f"Context:\n{context_block[:2000]}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.5,     # Low temperature = more factual
                repetition_penalty=1.2, # <--- THIS FIXES THE LOOPING BUG
                top_p=0.9
            )
        
        final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format sources
        source_str = "\n".join([f"📄 {name}" for name in set(retrieved_names)])
        
        return final_answer, source_str

    except Exception as e:
        return f"Error during generation: {str(e)}", ""

# --- 3. INTERFACE ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Medical Knowledge Bot (RAG)")
    gr.Markdown(f"Searching across **{len(documents)}** medical files.")
    
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Textbox(label="Ask a Question", placeholder="e.g., What are the symptoms of Asthma?", lines=2)
            btn = gr.Button("Get Answer", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=1):
            out_src = gr.Textbox(label="Source Documents", interactive=False, lines=10)
        with gr.Column(scale=3):
            out_ans = gr.Textbox(label="AI Answer", interactive=False, lines=10)
        
    btn.click(get_response, inputs=inp, outputs=[out_ans, out_src])

demo.launch()