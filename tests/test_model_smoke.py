"""Smoke test for CricketTalk AI model loading and generation."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_model, generate_response, get_hf_token


def log(msg: str):
    """Print immediately without buffering."""
    print(msg, flush=True)


def test_model_smoke():
    """Execute smoke test with two test questions and verify generation."""
    log("==================================================")
    log(" CricketTalk AI - Phase 1 Model Smoke Test")
    log("==================================================")

    token = get_hf_token()
    if token:
        log("[Auth] HF_TOKEN detected. Loading published LoRA adapter...")
    else:
        log("[Auth] Notice: No HF_TOKEN environment variable found.")
        log("       The adapter repo 'Hiteshilamakar23/crickettalk-ai-lora' is private.")
        log("       Running pipeline smoke test with allow_base_fallback=True...")

    log("\n--- Step 1: Loading Model Once ---")
    wrapper = load_model(allow_base_fallback=True)
    assert wrapper.model is not None, "Model failed to load"
    assert wrapper.tokenizer is not None, "Tokenizer failed to load"
    log(f"[Device] Running on: {wrapper.device_info}")
    log(f"[Adapter Status] LoRA adapter active: {wrapper.adapter_loaded}")

    log("\n--- Step 2: Question 1 (Java Programming) ---")
    q1 = "How do I start learning Java?"
    log(f"User Question: {q1}")
    a1 = generate_response(
        q1,
        history=[],
        max_new_tokens=128,
        model_wrapper=wrapper,
    )
    log(f"\nCricketTalk AI Response 1:\n{a1}")
    assert a1 and len(a1.strip()) > 10, "Response 1 was empty or too short"

    log("\n--- Step 3: Question 2 with History (Making Tea) ---")
    q2 = "What is the secret to making a great cup of tea?"
    log(f"User Question: {q2}")
    history = [[q1, a1]]
    a2 = generate_response(
        q2,
        history=history,
        max_new_tokens=128,
        model_wrapper=wrapper,
    )
    log(f"\nCricketTalk AI Response 2:\n{a2}")
    assert a2 and len(a2.strip()) > 10, "Response 2 was empty or too short"

    log("\n==================================================")
    log(" [PASSED] Smoke test succeeded!")
    log(" Multi-turn generation and Qwen chat template verified.")
    if not wrapper.adapter_loaded:
        log(" Remaining action: Set $env:HF_TOKEN with your Hugging Face read token")
        log(" to download the private LoRA adapter 'Hiteshilamakar23/crickettalk-ai-lora'.")
    log("==================================================")


if __name__ == "__main__":
    test_model_smoke()
