# CricketTalk AI — Project Audit Summary & Verification Record

**Document Date:** October 25, 2026  
**Auditor:** Senior Machine Learning & Documentation Specialist  
**Project Name:** CricketTalk AI: Fine-Tuning a Small Language Model for Cricket Commentary-Style Responses  
**Author / Student:** Hitesh Ilamkar  
**Target Repository:** `https://github.com/hitesh358-str/CricketTalkAi_-finetuned-model`  
**Colab Notebook:** `https://colab.research.google.com/drive/1UxDdYKaS7W73orXmXE25Z5PGlXW0ptgg?usp=sharing`  
**Published Adapter:** `https://huggingface.co/Hiteshilamakar23/crickettalk-ai-lora` (Private Access)

---

## 1. Verified Facts & Empirical Baseline

| Category | Verified Parameter / Finding | Primary Evidence Location |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` (498,431,872 total parameters, Causal LM) | Local Hugging Face cache `adapter_config.json`; Colab Cell 98 |
| **LoRA Configuration** | Rank $r=8$, Alpha $lpha=16$, Dropout 0.05, Target Modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `~/.cache/huggingface/hub/models--Hiteshilamakar23--crickettalk-ai-lora/snapshots/.../adapter_config.json` |
| **Trainable Parameters** | 4,399,104 trainable parameters (0.8826% of model parameters) | Notebook Cell 98 output (`trainable%: 0.8826`) |
| **Quantization (QLoRA)** | 4-bit NormalFloat4 (NF4) with double quantization, compute dtype `float16` | BitsAndBytesConfig in Colab Cell 98 |
| **Training Dataset** | 660 merged training examples (160 original base pairs + 500 unique scenarios) | `data/prepared/merged_train.jsonl` (Colab Cell 94/95) |
| **Validation Split** | 20 held-out examples | `data/splits/val_split.jsonl` (Colab Cell 95) |
| **Holdout Test Split** | 20 held-out examples across 5 disjoint scenarios × 4 prompt variations | `data/splits/test_split.jsonl` (Colab Cell 95, 100) |
| **Data Leakage Check** | Exactly 0 overlapping questions between train, validation, and test splits | Disjoint-set verification log (Colab Cell 95 output) |
| **Response Masking** | Token offset mapping with loss computed exclusively on assistant tokens (`labels = -100` on prompt/system tokens) | Colab Cell 98 custom collation script |
| **Training Hyperparams** | 3 Epochs, batch size 2, gradient accumulation 4 (effective batch 8), `paged_adamw_8bit`, learning rate 1e-4 with cosine decay (10 warmup steps), FP16 mixed precision | Colab Cell 98 `TrainingArguments` |
| **Hardware & Runtime** | Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM), PyTorch 2.11.0+cu130 | Colab Cell 97 output |
| **Published Adapter** | Uploaded to Hugging Face Hub as private repository `Hiteshilamakar23/crickettalk-ai-lora` | Colab Cell 104 output; `huggingFace snapshot.png` |
| **Gradio Application** | Interactive chatbot running on `127.0.0.1:7860`, connected to fine-tuned LoRA adapter | `app.py`; `Chatbot snapshot.png` |
| **Evaluation Scope** | 20 held-out test questions evaluated across 3 configurations: (A) Base Unprompted, (B) Prompted Base, (C) Fine-Tuned Model | Colab Cells 100–103; `model comparison.png` |

---

## 2. Quantitative Evaluation Summary (20 Held-Out Questions)

| Evaluated Metric | System A: Base Unprompted | System B: Prompted Base | System C: Fine-Tuned Adapter |
| :--- | :---: | :---: | :---: |
| **Cricket Metaphor Frequency** | 20.0% (4 / 20) | 30.0% (6 / 20) | **75.0% (15 / 20)** |
| **Clean Termination Rate (No Truncation)** | 50.0% (45% in audit) | 10.0% (2 / 20) | **100.0% (20 / 20)** |
| **Severe Truncation Rate (Hit 256 Max Tokens)** | 55.0% (11 / 20) | 90.0% (18 / 20) | **0.0% (0 / 20)** |
| **Average Response Length (Tokens)** | 229.0 tokens | 242.7 tokens | **62.7 tokens** |
| **Average Response Length (Characters)** | 1075.4 chars | 1127.4 chars | **312.6 chars** |
| **Qualitative Style Score (0–4 Scale)** | 0.00 / 4 (0.0%) | 2.00 / 4 (50.0%) | **4.00 / 4 (100.0%)** |
| **Factual Correctness (0–4 Scale)** | 4.00 / 4 (100.0%) | 4.00 / 4 (100.0%) | **4.00 / 4 (100.0%)** |
| **Helpfulness Score (0–4 Scale)** | 2.45 / 4 (61.25%) | 1.20 / 4 (30.0%) | **4.00 / 4 (100.0%)** |
| **Mean Rubric Score (0–4 Scale)** | 2.15 / 4 (53.75%) | 2.40 / 4 (60.0%) | **4.00 / 4 (100.0%)** |
| **Normalized Quality Percentage** | 53.75% | 60.0% | **100.0%** |

---

## 3. Findings Supported by Screenshot Evidence

1. **`Chatbot snapshot.png`**:
   - Confirms the active Gradio web application running on `127.0.0.1:7860`.
   - Demonstrates multi-turn conversational responses to everyday questions ("How do I start learning Java?" and "how to make cup of tea") with embedded cricket commentary terminology ("programming innings", "crease", "match is on").
2. **`huggingFace snapshot.png`**:
   - Confirms publication of the private model adapter at `https://huggingface.co/Hiteshilamakar23/crickettalk-ai-lora`.
   - Verifies model lineage: Base `Qwen/Qwen2.5-0.5B`, Fine-tuned `Qwen/Qwen2.5-0.5B-Instruct`.
   - Documents active download count (4 downloads) and parameter-efficient LoRA tags.
3. **`Trained adapter.png`**:
   - Documents Google Drive directory `.../CricketTalkAI/final_adapter_v2`.
   - Contains `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `README.md`, `tokenizer_config.json`, and `tokenizer.json`.
4. **`model comparison.png`**:
   - Shows the live Google Colab execution table comparing Base, Prompted Base, and Fine-Tuned Model across 5 general test questions.
   - Confirms exact prompt strings, model outputs, and output artifact paths (`quick_5_question_comparison.json`, `.csv`).

---

## 4. Key Limitations and Threats to Validity

1. **Regex Lexicon Boundary**: The 75.0% cricket metaphor frequency is calculated using a rule-based regex lexicon matching standard cricket terms (`wicket, pitch, innings, bowler, batter, boundary, umpire, fielder, over, run, six, stumps, defense`). Subtle colloquial commentary phrasing that did not use these explicit terms was not captured by the regex metric.
2. **Sample Size for Formal Evaluation**: Comparative evaluation was performed on 20 held-out test questions (5 scenarios × 4 prompt variations). While disjoint from training, a broader benchmark of 100+ questions would provide stronger statistical confidence.
3. **Prompted Base Model Verbosity**: The prompted base model suffered a 90% truncation rate when capped at 256 tokens, indicating that zero-shot prompting leads to verbose rambling, whereas fine-tuning teaches concise delivery.
4. **Adapter Access Status**: The Hugging Face repository remains **private**. Collaborators or graders must be granted read permissions or use an authorized `HF_TOKEN` to pull weights directly.

---

## 5. Artifacts and Deliverables Produced in `D:\CricketTalkAi_Submission_Report\`

- `CricketTalkAI_Submission_Report_Hitesh_Ilamkar.pdf` — Two-page executive assignment submission PDF.
- `CricketTalkAI_Technical_Evidence_Report_Hitesh_Ilamkar.pdf` — Comprehensive multi-page technical report with code, tables, and screenshots.
- `CricketTalkAI_Interactive_Report.html` — Self-contained interactive HTML presentation dashboard with offline-compatible relative assets.
- `evaluation_results.csv` — Machine-readable evaluation dataset containing all 60 test instances with individual metrics.
- `AUDIT_SUMMARY.md` — This technical verification document.
- `assets/` — Directory containing high-resolution comparison charts, pipeline diagrams, and authentic project screenshots.

*Zero modifications were made to the original repository codebase `D:\CricketTalkAi\`.*
