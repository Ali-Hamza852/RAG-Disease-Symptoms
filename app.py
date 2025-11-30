import gradio as gr
import faiss
import json
import torch
import os
from sentence_transformers import SentenceTransformer
# We use AutoModelForSeq2SeqLM because T5 is a Sequence-to-Sequence model (better for Q&A)
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
# We switch to FLAN-T5-BASE which is much better at following instructions than distilgpt2
print("🤖 Loading Instruction Model (Flan-T5)...")
model_name = "google/flan-t5-base" 

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Seq2SeqLM is required for T5 models
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. LOGIC ---
def get_response(user_query):
    if index is None or not documents:
        return "Error: System not ready. Please check if files were uploaded correctly.", ""

    try:
        # A. Search (Retrieval)
        # Normalize query because we used normalize_embeddings=True in Kaggle
        query_vector = embedder.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Find closest 3 matches
        distances, indices = index.search(query_vector, 3)
        
        # B. Retrieve Text
        retrieved_content = []
        retrieved_names = []
        
        for idx in indices[0]:
            if 0 <= idx < len(documents):
                # Clean up text to save tokens
                text = documents[idx][:1000] # Limit chunk size
                retrieved_content.append(text)
                retrieved_names.append(sources[idx])

        # C. Generate Answer (Generation)
        # T5 likes simple instructions.
        context_block = "\n".join(retrieved_content)
        
        # This prompt is specific for T5/Flan models
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context_block}\n\nQuestion: {user_query}"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,   # Allow longer answers
                do_sample=True,       # Add a little creativity
                temperature=0.6,      # Keep it factual
                top_p=0.9
            )
        
        final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format sources for display
        source_str = "\n".join([f"📄 {name}" for name in set(retrieved_names)])
        
        return final_answer, source_str

    except Exception as e:
        return f"Error during generation: {str(e)}", ""

# --- 3. INTERFACE ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Medical Knowledge Bot (RAG)")
    gr.Markdown(f"Searching across **{len(documents)}** medical files using **Google Flan-T5**.")
    
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Textbox(label="Ask a Question", placeholder="e.g., What are the symptoms of heart failure?", lines=2)
            btn = gr.Button("Get Answer", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=1):
            out_src = gr.Textbox(label="Source Documents", interactive=False, lines=10)
        with gr.Column(scale=3):
            out_ans = gr.Textbox(label="AI Answer", interactive=False, lines=10)
        
    btn.click(get_response, inputs=inp, outputs=[out_ans, out_src])

demo.launch()