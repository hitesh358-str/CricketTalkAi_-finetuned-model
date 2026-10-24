"""CricketTalk AI - Web Chatbot Application.

Interactive Gradio interface for cricket commentary-style conversational AI.
Connected to Qwen2.5-0.5B-Instruct fine-tuned with the published LoRA adapter.
"""

from typing import Any, List, Optional, Union
import gradio as gr
from src.model import CricketTalkModel, generate_response, load_model

# Cached global singleton model wrapper and startup status
_MODEL_WRAPPER: Optional[CricketTalkModel] = None
_MODEL_INIT_ERROR: Optional[str] = None


def get_model() -> CricketTalkModel:
    """Retrieve or initialize the model singleton loaded once at application start.

    Returns:
        CricketTalkModel instance.

    Raises:
        RuntimeError: If model or LoRA adapter fails to load.
    """
    global _MODEL_WRAPPER, _MODEL_INIT_ERROR
    if _MODEL_WRAPPER is None and _MODEL_INIT_ERROR is None:
        try:
            print("Loading CricketTalk AI model and LoRA adapter into memory...")
            _MODEL_WRAPPER = load_model(allow_base_fallback=False)
            if not _MODEL_WRAPPER.adapter_loaded:
                raise RuntimeError(
                    "LoRA adapter 'Hiteshilamakar23/crickettalk-ai-lora' could not be attached."
                )
            print(f"Model ready on {_MODEL_WRAPPER.device_info} (LoRA active: True).")
        except Exception as exc:
            _MODEL_INIT_ERROR = str(exc)
            print(f"Model initialization error: {_MODEL_INIT_ERROR}")
            raise exc

    if _MODEL_INIT_ERROR:
        raise RuntimeError(_MODEL_INIT_ERROR)

    return _MODEL_WRAPPER


def respond(message: Union[str, dict], history: list) -> str:
    """Generate a response for the chatbot using the fine-tuned CricketTalk AI model.

    Args:
        message: The user's input message.
        history: The conversation history (multi-turn).

    Returns:
        A response string generated in cricket commentary style.
    """
    text = ""
    if isinstance(message, dict):
        text = str(message.get("text", "")).strip()
    elif message is not None:
        text = str(message).strip()

    if not text:
        return "Please enter a valid question or topic to get started!"

    # Retrieve pre-loaded model
    try:
        model_wrapper = get_model()
    except Exception as e:
        return (
            f"[CricketTalk AI Unavailable] Could not load model: {e}\n\n"
            "Please ensure your Hugging Face read token is configured if accessing a private adapter:\n"
            "  PowerShell: $env:HF_TOKEN = 'your_token'\n"
            "  Or add HF_TOKEN to a .env file in the project root."
        )

    try:
        # Generate answer using Qwen chat template and LoRA adapter
        return generate_response(
            message=text,
            history=history,
            model_wrapper=model_wrapper,
        )
    except Exception as err:
        return f"An error occurred while generating the commentary response: {err}"


def build_app() -> gr.ChatInterface:
    """Build and configure the Gradio ChatInterface application.

    Returns:
        Configured Gradio ChatInterface instance.
    """
    examples = [
        "How do I start learning Java?",
        "What are some effective tips for time management?",
        "How do I make a great cup of tea?",
    ]

    demo = gr.ChatInterface(
        fn=respond,
        title="CricketTalk AI",
        description=(
            "CricketTalk AI answers everyday questions in an energetic, "
            "vivid cricket commentary style! Powered by Qwen2.5-0.5B-Instruct "
            "and fine-tuned with the published LoRA adapter."
        ),
        examples=examples,
    )
    return demo


def main() -> None:
    """Main application entry point."""
    print("Pre-loading CricketTalk AI model once at startup...")
    try:
        get_model()
    except Exception as e:
        print(f"Startup notice: Model will retry on first request ({e})")

    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
