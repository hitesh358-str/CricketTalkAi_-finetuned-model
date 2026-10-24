# CricketTalk AI

CricketTalk AI is a lightweight language model fine-tuned to answer general questions in an energetic, engaging cricket commentary style. The project aims to fine-tune a small open-weight language model using parameter-efficient fine-tuning (PEFT/LoRA), benchmark and evaluate it against two baselines, and deliver an interactive web chatbot interface.

## Planned Workflow

1. **Environment Setup & Dependency Resolution**: Configure the Python virtual environment and install compatible versions of Transformers, TRL, PEFT, Gradio, and supporting libraries.
2. **Data Pipeline**:
   - Ingest raw commentary transcripts and QA datasets into `data/raw/`.
   - Clean, format, and tokenize training pairs into `data/processed/`.
   - Assemble holdout evaluation sets in `data/test/`.
3. **Exploratory Notebooks**: Experiment with prompts and model behavior in `notebooks/`.
4. **Model Fine-Tuning**: Fine-tune a small open-weight LLM via PEFT/LoRA using `trl` and `accelerate` with modular code in `src/`.
5. **Evaluation & Benchmarking**: Compare fine-tuned model outputs against two baselines (e.g., zero-shot base model and general instruction-tuned model) across commentary style and factual consistency; document results in `reports/`.
6. **Web Chatbot Interface**: Develop and launch an interactive Gradio web application in `app.py`.
7. **Testing**: Validate pipelines and endpoints using `pytest` under `tests/`.

## Project Structure

```
crickettalk-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
├── notebooks/
├── src/
│   └── __init__.py
├── tests/
├── reports/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
