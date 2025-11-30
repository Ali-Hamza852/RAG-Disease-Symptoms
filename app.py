import gradio as gr
import faiss
import json
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. SETUP ---
print("🔄 System Starting...")
device = "cpu" # HF Spaces free tier uses CPU

# Load Embedding Model
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

# Load LLM (Chat Model)
print("🤖 Loading Chat Model...")
# Check if the uploaded config folder exists, otherwise download default
model_path = "llm_config" if os.path.exists("llm_config") else "distilgpt2"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
except Exception as e:
    print(f"⚠️ Could not load custom model, falling back to default. Error: {e}")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. LOGIC ---
def get_response(user_query):
    if index is None or not documents:
        return "Error: System not ready. Please check if files were uploaded correctly.", ""

    try:
        # A. Search
        # Normalize query because we used normalize_embeddings=True in Kaggle
        query_vector = embedder.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Find closest 3 matches
        distances, indices = index.search(query_vector, 3)
        
        # B. Retrieve Text
        retrieved_content = []
        retrieved_names = []
        
        for idx in indices[0]:
            if 0 <= idx < len(documents):
                retrieved_content.append(documents[idx])
                retrieved_names.append(sources[idx])

        # C. Generate Answer
        context_block = "\n---\n".join(retrieved_content)
        # We limit context to 1500 chars to fit in small models
        prompt = f"Context information:\n{context_block[:1500]}\n\nQuestion: {user_query}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up prompt from response
        final_answer = full_response.replace(prompt, "").strip()
        
        # Format sources
        source_str = "\n".join([f"📄 {name}" for name in set(retrieved_names)])
        
        return final_answer, source_str

    except Exception as e:
        return f"Error during generation: {str(e)}", ""

# --- 3. INTERFACE ---
# REMOVED THE THEME ARGUMENT TO FIX THE ERROR
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Medical Knowledge Bot")
    gr.Markdown(f"RAG System searching across **{len(documents)}** dataset files.")
    
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Textbox(label="Ask a Question", placeholder="Type your medical question here...", lines=2)
            btn = gr.Button("Get Answer", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=1):
            out_src = gr.Textbox(label="Source Files Used", interactive=False, lines=10)
        with gr.Column(scale=3):
            out_ans = gr.Textbox(label="AI Answer", interactive=False, lines=10)
        
    btn.click(get_response, inputs=inp, outputs=[out_ans, out_src])

demo.launch()