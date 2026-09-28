"""Encode the 899 distractor texts (text_input: question + correct answer + distractor) with every encoder used in the
paper, and the 2,587 misconception descriptions with the two retrievers. Results are cached in cache/embeddings/.

  SciBERT      allenai/scibert_scivocab_uncased, CLS vector, max 128 tokens, L2-normalised
  others       sentence-transformers, L2-normalised; prefixes follow each model card for symmetric / clustering use
               (e5: "query: ", nomic: "clustering: "; no instruction for the rest)
  Qwen3-4B/8B  run in bfloat16; on an 8 GB GPU they fall back to CPU (about 15 / 30 minutes)
  bank_*       misconception descriptions encoded with MiniLM and Qwen3-Embedding-0.6B (the two retrievers)

gte-large-en-v1.5 was tried but excluded: its remote modelling code is incompatible with transformers >= 5 and
produces collapsed embeddings (mean pairwise cosine 0.99).

Usage: python scripts/01_encode.py [encoder names ...]     (default: all; cached encoders are skipped)
"""
import sys, io, time, zipfile, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from config import DISTRACTORS, MISC_ZIP, EMB

texts = pd.read_parquet(DISTRACTORS).reset_index(drop=True).text_input.fillna("").astype(str).tolist()
with zipfile.ZipFile(MISC_ZIP) as z:
    bank = pd.read_csv(io.BytesIO(z.read("misconception_mapping.csv"))).MisconceptionName.tolist()
GPU = "cuda" if torch.cuda.is_available() else "cpu"
BIG = GPU if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9 else "cpu"

ST = {  # name: (hf id, prefix, device, kwargs)
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", GPU, {}),
    "all-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", "", GPU, {}),
    "bge-large-en-v1.5": ("BAAI/bge-large-en-v1.5", "", GPU, {}),
    "e5-large-v2": ("intfloat/e5-large-v2", "query: ", GPU, {}),
    "mxbai-embed-large-v1": ("mixedbread-ai/mxbai-embed-large-v1", "", GPU, {}),
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5", "clustering: ", GPU, {"trust_remote_code": True}),
    "Qwen3-Embedding-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "", GPU, {}),
    "Qwen3-Embedding-4B": ("Qwen/Qwen3-Embedding-4B", "", BIG, {"model_kwargs": {"torch_dtype": torch.bfloat16}}),
    "Qwen3-Embedding-8B": ("Qwen/Qwen3-Embedding-8B", "", BIG, {"model_kwargs": {"torch_dtype": torch.bfloat16}}),
}
BANK = {"bank_minilm": "all-MiniLM-L6-v2", "bank_qwen3-0.6b": "Qwen3-Embedding-0.6B"}
only = set(sys.argv[1:])

def scibert(tx):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    mdl = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(GPU).eval(); out = []
    with torch.no_grad():
        for i in range(0, len(tx), 32):
            enc = tok(tx[i:i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt").to(GPU)
            out.append(mdl(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
    E = np.vstack(out).astype(np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

def st(name, tx):
    from sentence_transformers import SentenceTransformer
    hf, prefix, dev, kw = ST[name]
    m = SentenceTransformer(hf, device=dev, **kw); m.max_seq_length = min(m.max_seq_length or 512, 512)
    E = m.encode([prefix + t for t in tx], batch_size=4 if dev == "cpu" else 32, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    del m; gc.collect(); torch.cuda.empty_cache()
    return E

jobs = [("SciBERT", lambda: scibert(texts))] + [(n, (lambda n=n: st(n, texts))) for n in ST] + \
       [(b, (lambda n=n: st(n, bank))) for b, n in BANK.items()]
for name, fn in jobs:
    p = EMB / f"{name}.npy"
    if (only and name not in only) or p.exists():
        continue
    t0 = time.time(); E = fn(); assert np.isfinite(E).all()
    np.save(p, E); print(f"[done] {name} {E.shape} in {time.time() - t0:.0f}s", flush=True)
print("embeddings in", EMB)
