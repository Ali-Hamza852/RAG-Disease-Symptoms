
from pathlib import Path
import gradio as gr
import json, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleRAG:
    def __init__(self, dataset_path, embedding_model="all-MiniLM-L6-v2", llm_model="distilgpt2", device="cpu"):
        self.dataset_path = Path(dataset_path)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(embedding_model, device=self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator = AutoModelForCausalLM.from_pretrained(llm_model).to(self.device)
            self.generator.eval()
        except:
            self.generator, self.tokenizer = None, None
        self.documents, self.doc_sources, self.embeddings, self.index = [], [], None, None

    def _find_json_files(self):
        roots = []
        finished = self.dataset_path / "Finished"
        samples = self.dataset_path / "samples"
        if finished.exists(): roots.append(finished)
        if samples.exists(): roots.append(samples)
        if not roots: roots = [self.dataset_path]
        files = []
        for r in roots:
            files.extend(list(r.rglob("*.json")))
        return sorted(list({p.resolve(): p for p in files}.values()), key=lambda p: str(p))

    def load_data(self, max_files=2000):
        json_files = self._find_json_files()
        docs, srcs, count = [], [], 0
        for jf in json_files:
            if count >= max_files: break
            try:
                with open(jf,"r",encoding="utf-8",errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    parts = []
                    for k,v in data.items():
                        if "input" in k.lower(): parts.append(f"{k}: {v}")
                    if not parts:
                        for k,v in data.items():
                            parts.append(f"{k}: {v}" if isinstance(v,(str,int,float)) else f"{k}: {str(v)[:300]}")
                    text = "\n".join(parts).strip()
                else:
                    text = str(data)
                if text: docs.append(text); srcs.append(str(jf)); count+=1
            except: continue
        self.documents, self.doc_sources = docs, srcs
        return len(docs)

    def _chunk(self, text, size=300, overlap=50):
        words = text.split()
        if len(words)<=size: return [text]
        chunks=[]
        step=size-overlap
        for i in range(0,len(words),step):
            chunks.append(" ".join(words[i:i+size]))
            if i+size>=len(words): break
        return chunks

    def build_index(self, chunk_size=300, chunk_overlap=50, batch_size=64):
        if not self.documents: raise RuntimeError("No documents loaded.")
        chunks, chunk_sources = [], []
        for doc,src in zip(self.documents,self.doc_sources):
            parts = self._chunk(doc, chunk_size, chunk_overlap)
            chunks.extend(parts)
            chunk_sources.extend([src]*len(parts))
        self.documents = chunks
        self.doc_sources = chunk_sources
        self.embeddings = self.encoder.encode(chunks, convert_to_numpy=True, batch_size=batch_size).astype("float32")
        dim=self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.embeddings)
        self.index = index

    def save_artifacts(self, out_dir="./hf_model_dir"):
        out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index,str(out_dir/"faiss_index.idx"))
            np.save(out_dir/"embeddings.npy", self.embeddings)
            with open(out_dir/"doc_sources.json","w",encoding="utf-8") as f: json.dump(self.doc_sources,f,ensure_ascii=False,indent=2)
        if getattr(self,"tokenizer",None): self.tokenizer.save_pretrained(out_dir)
        if getattr(self,"generator",None): self.generator.save_pretrained(out_dir)

    def retrieve(self, query, top_k=5):
        if self.index is None: raise RuntimeError("Index not built.")
        q_emb = self.encoder.encode([query], convert_to_numpy=True)[0].astype("float32")
        D,I=self.index.search(np.array([q_emb]),top_k)
        return [{"text":self.documents[idx],"source":self.doc_sources[idx],"distance":float(dist),"similarity":1.0/(1.0+float(dist))} for idx,dist in zip(I[0],D[0])]

    def generate(self, query, docs, max_length=150):
        if self.generator is None or self.tokenizer is None:
            ctx="\n\n---\n\n".join([f'[{d["source"]}] {d["text"][:800]}' for d in docs[:5]])
            return "Context-only (no LLM loaded):\n\n"+ctx
        context="\n\n".join([d["text"] for d in docs[:3]])
        prompt=f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        inputs=self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.generator.device)
        with torch.no_grad():
            out=self.generator.generate(**inputs,max_length=min(inputs['input_ids'].shape[1]+max_length,1024),do_sample=True,temperature=0.7,top_p=0.95,pad_token_id=self.tokenizer.eos_token_id)
        decoded=self.tokenizer.decode(out[0],skip_special_tokens=True)
        return decoded.replace(prompt,"").strip()

    def ask(self, question, top_k=5):
        docs=self.retrieve(question, top_k)
        ans=self.generate(question, docs)
        return {"question":question,"answer":ans,"sources":docs}

DATA_ROOT = Path("rag_data") / "mimic-iv-ext-direct-1.0.0"
rag = SimpleRAG(dataset_path=str(DATA_ROOT))
if not Path("hf_model_dir").exists():
    rag.load_data(max_files=2000)
    rag.build_index()
    rag.save_artifacts("hf_model_dir")

def respond(q):
    res = rag.ask(q, top_k=5)
    srcs=[]
    for i,s in enumerate(res["sources"]):
        snippet=s["text"][:400].replace("\n"," ")
        srcs.append(f"{i+1}. {Path(s['source']).name} - dist:{s['distance']:.3f}\n{snippet}...")
    return res["answer"], "\n\n".join(srcs)

with gr.Blocks() as demo:
    gr.Markdown("# SimpleRAG — Ask the dataset")
    with gr.Row():
        q = gr.Textbox(placeholder="Ask a question...", label="Question", lines=2)
        btn = gr.Button("Ask")
    out_ans = gr.Textbox(label="Answer", lines=8)
    out_src = gr.Textbox(label="Top Sources", lines=10)
    btn.click(respond, inputs=q, outputs=[out_ans, out_src])

demo.launch()
