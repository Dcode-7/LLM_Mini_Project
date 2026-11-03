# College Support Assistant

A RAG-based student helpdesk for **Northbridge College of Engineering & Technology** (a fictional sample college) with a fine-tuned LLM. It combines:

- **Experiment 7**: a RAG app with input validation, error handling, logging, response-time measurement and endpoint tests, using a swappable LLM backend (cloud API or Ollama). It is available as a FastAPI service and as a Streamlit web app.
- **Lecture 4 (PEFT / LoRA / QLoRA)**: `Qwen2.5-0.5B-Instruct` fine-tuned on the college knowledge base with **LoRA** and **QLoRA**, evaluated against the base model. The results are shown in the app.

Every answer also includes a **helpdesk ticket** (Department / Category / Priority / Action), as in the lecture's *University Helpdesk Case*:

```
Input:  "My semester fee was deducted but the portal still shows unpaid."
Answer: Online payments can take up to 48 hours to reconcile ... email accounts@northbridge.edu with the UTR number ...
Ticket: Department: Accounts | Category: Payment Reconciliation | Priority: High | Action: Verify transaction reference
```

## Project structure

```
streamlit_app.py            Streamlit web app (hosted): Assistant, Knowledge base (PDF), Fine-tuning pages
kb_view.py                  Knowledge-base page: every PDF processing step + retrieval explorer
finetuning_view.py          Fine-tuning page: LoRA vs QLoRA, loss curves, evaluation, sample answers, calculator
pdf_ingest.py               PDF ingestion: extract -> clean -> detect sections/topics -> chunk -> parse metadata
support_rag.py              Index over PDF chunks + TF-IDF retriever + prompt builder + LLM backends
app.py                      FastAPI service: /ask, /health, /docs (+ HTML chat at /) for API use and Experiment 7 tests
static/index.html           Chat page served by FastAPI
data/college_handbook.pdf   THE KNOWLEDGE BASE: 23-page Student Helpdesk Handbook, 15 sections, 121 topics
data/build_handbook_pdf.py  Generates the handbook PDF from the authored source data (reportlab)
data/college_kb.json        Authored source data for the PDF (and for fine-tuning questions); not read by the app
tests/                      30 tests: PDF ingestion, API endpoints, Streamlit pages
requirements.txt            Runtime dependencies (what Streamlit Cloud installs)
requirements-finetune.txt   torch / transformers / peft / bitsandbytes for fine-tuning
.streamlit/                 Theme config + secrets template (real secrets.toml is git-ignored)
finetune/
  prepare_dataset.py        Knowledge base -> data/train.jsonl (148) / data/val.jsonl (66)
  train_lora.py             LoRA (default) or QLoRA (--qlora: 4-bit NF4 + double quant + paged optimizer)
  evaluate_lora.py          Base vs base+LoRA vs base+QLoRA on held-out questions
  lora_math.py              Parameter arithmetic from the lecture (256x, 128x, GB per precision)
  adapters/college-lora/    Trained LoRA adapter (17.6 MB)
  adapters/college-qlora/   Trained QLoRA adapter (17.6 MB)
  results/*.json            Training metrics, loss curves, evaluation and sample answers
```

## Architecture

```
college_handbook.pdf ──► pdf_ingest: extract (pypdf) ─► clean ─► sections/topics ─► 121 chunks + metadata
                                        │   (once at startup, cached)
                     Index: word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams) over the chunks
                                        │
Streamlit chat / FastAPI /ask ──► validation (1–1000 chars, not blank) ──► rate limit (30/min)
                                        │
                     Retrieve top-3 chunks (score = 0.6·word + 0.4·char cosine similarity)
                                        │
                  best score < MIN_SCORE? ── yes ──► "I don't have enough information…" (no LLM call)
                                        │ no
                 LLM backend (LLM_BACKEND):  extractive | cloud | ollama | lora (fine-tuned Qwen)
                                        │
            answer + grounded flag + ticket {department, category, priority, action} + sources + latency_ms
```

| `LLM_BACKEND` | Where the model runs | Notes |
|---|---|---|
## Knowledge base: how the PDF is processed

The knowledge base is a single document, `data/college_handbook.pdf`, a 23-page *Student Helpdesk Handbook*. `pdf_ingest.py` processes it at startup. The **Knowledge base (PDF)** page in the app shows every step on the real document: raw vs cleaned text per page, the chunk table, a retrieval explorer with scores, and the exact prompt sent to the LLM.

| Step | What happens | Result on the handbook |
|---|---|---|
| 1. Extract | `pypdf` pulls text from each page; PDF metadata (title, edition) is read | 23 pages, about 70 ms |
| 2. Clean | Running header, footer and page-number lines are removed; the cover and contents pages are detected and discarded | 3 noise lines removed per page; pages 1–2 skipped |
| 3. Structure | `Section N - …` headings and `[ID] …` topic headings are detected by pattern; line-wraps are joined | 15 sections, 121 topics |
| 4. Chunk + metadata | One chunk per topic, so a chunk never mixes two policies. Answers over 900 chars would be split with 150-char overlap. The printed *Also asked as* and *Routed to* lines become fields (alternate phrasings, department, category, priority, action), and each chunk keeps its section and page number | 121 chunks, 190–452 chars; all match the source exactly (tested) |
| 5. Index | Title (twice), alternate phrasings, category and answer are indexed with word and character TF-IDF | built in a few ms |
| 6. Retrieve | Top-3 chunks above the 0.17 threshold go into the prompt; below it the app says "not enough information" without calling the LLM | sources show `handbook p. N` |

**Why chunk by structure instead of by size?** Splitting the same text every 500 characters gives 87 chunks, and **77 of them mix two different topics**: a hostel fee chunk could carry half of the hostel application rules, and the routing line gets separated from its answer. The Knowledge base page shows this comparison.

To change the knowledge base, edit `data/college_kb.json` and run `python data/build_handbook_pdf.py`, or replace the PDF with one that follows the same layout. Run `python pdf_ingest.py` to see a processing report.

## LLM backends

| `LLM_BACKEND` | Where the model runs | Notes |
|---|---|---|
| `extractive` | nowhere (no LLM) | Returns the best-matching curated answer. Default; also serves as the mock backend. |
| `cloud` | provider's server | Any OpenAI-compatible API: Groq, Hugging Face router, OpenRouter… Needs `LLM_API_KEY`. |
| `ollama` | your computer | `ollama pull qwen2.5:0.5b`, then `ollama serve`. |
| `lora` | your computer | Qwen2.5-0.5B + the fine-tuned adapter in `finetune/adapters/`. Needs `requirements-finetune.txt`. |

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt

streamlit run streamlit_app.py        # web app      -> http://localhost:8501
uvicorn app:app --reload              # REST API     -> http://127.0.0.1:8000/docs
pytest -v                             # 30 tests
```

To chat with **your fine-tuned model** locally:

```powershell
pip install -r requirements-finetune.txt
$env:LLM_BACKEND="lora"; streamlit run streamlit_app.py
```

## Host on Streamlit Community Cloud

1. **Push the project to a GitHub repository.** `.env` and `.streamlit/secrets.toml` are git-ignored:
   ```powershell
   git init
   git add .
   git commit -m "College Support Assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/college-support-assistant.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app → Deploy a public app from GitHub**.
3. Choose your repo, branch `main`, and main file `streamlit_app.py`. Under **Advanced settings**, set Python to **3.11** and paste your secrets (format as in `.streamlit/secrets.toml.example`):
   ```toml
   LLM_BACKEND = "cloud"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "openai/gpt-oss-20b"
   LLM_API_KEY = "your-groq-key"
   ```
   Free Groq key: https://console.groq.com/keys. If you have no key, use `LLM_BACKEND = "extractive"`.
4. Click **Deploy**. The app will be live at `https://<your-app-name>.streamlit.app`.

Notes:
- Streamlit Cloud has no GPU and limited RAM, so the hosted chat uses the cloud API (or the extractive backend). The fine-tuned model runs locally with `LLM_BACKEND=lora`. The **Fine-tuning** page shows its training and evaluation results from the saved JSON files.
- Hosting the app in the cloud does not host the LLM there; the model is the provider's.
- Free apps sleep after a period of inactivity. The first visit then takes about 30 s to wake up.

## Fine-tuning (Lecture 4)

```bash
pip install -r requirements-finetune.txt
python finetune/prepare_dataset.py          # 148 train / 66 val chat examples
python finetune/train_lora.py               # LoRA:  bf16 frozen base, r=8, alpha=16, dropout=0.05
python finetune/train_lora.py --qlora       # QLoRA: 4-bit NF4 frozen base + LoRA (NVIDIA GPU)
python finetune/evaluate_lora.py            # base vs LoRA vs QLoRA -> finetune/results/eval.json
python finetune/lora_math.py                # 4096x4096,r=8 -> 256x ; 2048x2048,r=8 -> 128x
```

Each training example uses the **same retrieved-context prompt as the app**: RAG supplies current knowledge, LoRA supplies specialized behavior. The model learns to answer concisely with exact fees, dates and contacts, and to refuse off-topic questions. Validation uses paraphrases never seen in training plus new off-topic questions.

### Training results (RTX 3050 6 GB laptop GPU, 3 epochs)

| | LoRA | QLoRA |
|---|---|---|
| Frozen base stored as | bf16 | 4-bit NF4 + double quant |
| Base parameters | 494.0 M | 494.0 M |
| Trainable parameters (A, B) | 4.40 M (0.89 %) | 4.40 M (0.89 %) |
| Base-weight memory | 988 MB | 724 MB |
| Peak GPU memory while training | 9.3 GB (spilled past 6 GB VRAM into shared memory) | **5.4 GB (fits in VRAM)** |
| Training time | 201 s | 191 s |
| Final validation loss | 0.043 | 0.048 |
| Saved adapter | 17.6 MB | 17.6 MB |

Takeaways:
- About 0.9% of parameters were trained, but the full 494 M-parameter base model still exists and is still needed at inference.
- Model weight size ≠ training memory: the weights are 1 GB but LoRA training needed about 9 GB.
- QLoRA's 4-bit base brought training under the card's 6 GB at almost the same validation loss.
- Embeddings and the output layer stay unquantized, so the 4-bit base is 724 MB, not 250 MB.

### Evaluation (held-out set)

The validation set has 66 questions: 63 paraphrases never seen in training, plus 3 new off-topic questions. Every model gets the same retrieved context and uses greedy decoding.

| Model | Answer word-overlap F1 | Facts reproduced (numbers) | Off-topic refused | Time / answer |
|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct | 0.523 | 51 % | 3 / 3 | 2.2 s |
| Base + **LoRA** | **0.962** | **97 %** | 3 / 3 | 5.8 s |
| Base + **QLoRA** | **0.975** | **98 %** | 2 / 3 | 7.5 s |

What the numbers show:
- **Fine-tuning changed the behavior.** The base model often rambles ("To answer the student's question… we need to refer to the admissions policy…") or drops details. Both adapters give the college's exact answer, with fees, deadlines and contacts intact.
- **QLoRA nearly matches LoRA in accuracy**, despite its 4-bit base. It missed one of the three off-topic questions: for "Recommend a good movie" it answered with the mess-feedback entry. With only 3 off-topic samples this is a small test, so add more refusal examples before relying on it.
- **LoRA does not guarantee faster inference.** The unmerged adapter adds extra matrix multiplications, the fine-tuned answers are longer and complete, and the 4-bit base must be dequantized on the fly.
- **Caveat:** these scores are high partly because the retriever already places the right entry in the prompt. The adapter learned to use that context faithfully, which is the intended RAG + LoRA division of labor.

### Where the lecture concepts appear in the code

| Concept | Where |
|---|---|
| Frozen base, trainable A & B | `get_peft_model(...)`, `print_trainable_parameters()` |
| Rank `r`, alpha `α` (scale α/r), dropout | `--r`, `--alpha`, `--dropout` in `train_lora.py` |
| Target modules (architecture-specific) | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (Qwen2 names) |
| Quantized base (NF4, double quantization) | `BitsAndBytesConfig(load_in_4bit, nf4, double_quant)` |
| Storage precision ≠ compute precision | `bnb_4bit_compute_dtype=bfloat16` |
| Paged optimizers | `optim="paged_adamw_8bit"` with `--qlora` |
| What gets saved | Only the adapter (17.6 MB); base + adapter are needed for inference |

## Observation table (Experiment 7, local run, extractive backend)

| Test case | Expected result | Actual result | Pass/Fail |
|---|---|---|---|
| Health endpoint | API reports running | `200 {"status":"ok","backend":"extractive","kb_entries":121}` | Pass |
| Supported question | Grounded answer | "lost library book" → LIB-003 answer, ticket Library/Medium | Pass |
| Unsupported question | Insufficient information / fallback | "capital of Australia" → fallback message, `grounded:false` | Pass |
| Empty question | Validation error | `422 {"error":"invalid_input","detail":["Question must not be empty"]}`; Streamlit shows a warning | Pass |
| Too-long question (1001 chars) | Rejected | `422 … "Question is too long (maximum 1000 characters)"`; Streamlit input capped at 1000 | Pass |
| Backend unavailable (cloud, no key) | Controlled error, no secrets | `503 {"error":"backend_unavailable",…}`; reason logged server-side only | Pass |
| Response-time measurement | Latency recorded | `latency_ms` in every response and log line (~4 ms extractive) | Pass |

**Selected backend:** ☑ Cloud-hosted API (Streamlit Cloud deployment) · ☑ Ollama / fine-tuned LoRA (local)

## Discussion questions: short answers

1. **Purpose of FastAPI.** It exposes the RAG pipeline as HTTP endpoints that other applications can call. It also gives request validation and auto-generated `/docs`. Streamlit provides the human-facing UI.
2. **Local vs cloud LLM.** A local LLM (Ollama, or our LoRA model) runs on your own hardware: no per-request cost, data stays local, but speed depends on your machine. A cloud API runs on the provider's servers: fast with no GPU needed, but it needs a key, has quotas and sends data over the internet.
3. **Does hosting the app in the cloud host the LLM?** No. Streamlit Cloud runs only the app; the model is still served by the provider.
4. **Why not hard-code API keys?** Code is shared and versioned, so a committed key leaks to everyone with repo access (and to bots that scan GitHub). Use `.env` or `st.secrets`, and keep both out of git.
5. **Input validation and max length.** Validation rejects bad input early with a clear error. A length cap bounds token cost and latency, and reduces abuse and prompt-stuffing.
6. **Health endpoint: purpose and limit.** It shows the process is alive. It does not prove the LLM provider, quota or answer quality are fine.
7. **Why load the model and index once?** Loading per request adds seconds of latency and wastes memory. FastAPI uses `lifespan` and Streamlit uses `@st.cache_resource` to load them once.
8. **What if the answer is not in the knowledge base?** Say so and point to a human contact. This app does that by a similarity threshold and by the prompt instruction, which the fine-tuned model also learned.
9. **Two reasons a cloud request fails.** Invalid or expired key (401), exhausted quota or rate limit (429), network timeout, or a retired or misspelled model name (404).
10. **Is it fully production-ready?** No. A real public service would also need authentication, HTTPS with a custom domain, a distributed rate limiter, monitoring and alerting, managed secrets, load testing, systematic model evaluation, and a backup/recovery plan.

## Disclaimer

Northbridge College and all fees, dates and contacts in `data/college_kb.json` are fictional sample data. Replace them with real information before any real use, and do not send confidential student data to third-party APIs.
