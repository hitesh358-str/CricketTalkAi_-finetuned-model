"""Model loading and inference module for CricketTalk AI.

Loads Qwen2.5-0.5B-Instruct base model with the published LoRA adapter
(Hiteshilamakar23/crickettalk-ai-lora). Supports CUDA (with optional 4-bit quantization
when hardware allows) and CPU fallback.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

# Disable hf-xet on Windows if present to prevent hanging on certain network configurations
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_MODEL_ID = "Hiteshilamakar23/crickettalk-ai-lora"
DEFAULT_SYSTEM_PROMPT = (
    "You are CricketTalk AI, an expert assistant who answers general questions "
    "in an authentic, engaging cricket commentary style while keeping answers factually accurate, "
    "clear, and genuinely useful."
)


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from environment or local .env file if available.

    Never logs or exposes the token value.
    """
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()

    # Fallback: check .env file in workspace root
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass

    return None


def can_use_4bit_quantization() -> bool:
    """Check if 4-bit quantization via bitsandbytes is supported on current hardware."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        # bitsandbytes 4-bit generally requires compute capability >= 7.0
        if major < 7:
            return False
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig  # noqa: F401

        return True
    except Exception:
        return False


def resolve_local_adapter_snapshot(adapter_id: str) -> Optional[str]:
    """Find local cached snapshot directory for the adapter if available."""
    if os.path.isdir(adapter_id):
        return adapter_id

    hub_folder = "models--" + adapter_id.replace("/", "--")
    hub_cache = os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub", hub_folder, "snapshots")
    )
    if os.path.isdir(hub_cache):
        snapshots = [
            os.path.join(hub_cache, d)
            for d in os.listdir(hub_cache)
            if os.path.isdir(os.path.join(hub_cache, d))
        ]
        if snapshots:
            snapshots.sort(key=lambda s: os.path.getmtime(s), reverse=True)
            for s in snapshots:
                if os.path.exists(os.path.join(s, "adapter_config.json")):
                    return s
    return None


class CricketTalkModel:
    """Singleton wrapper for CricketTalk AI tokenizer and model."""

    _instance: Optional["CricketTalkModel"] = None

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        adapter_model_id: str = ADAPTER_MODEL_ID,
        allow_base_fallback: bool = False,
    ):
        self.base_model_id = base_model_id
        self.adapter_model_id = adapter_model_id
        self.allow_base_fallback = allow_base_fallback
        self.token = get_hf_token()
        self.adapter_loaded = False
        self.tokenizer, self.model, self.device_info = self._load()

    def _load(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load tokenizer, base model, and LoRA adapter."""
        print(f"Loading tokenizer from '{self.base_model_id}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                token=self.token,
                trust_remote_code=True,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_4bit = can_use_4bit_quantization()
        model = None
        device_info: Dict[str, Any] = {}

        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                print("CUDA hardware compatible: Loading base model in 4-bit quantization...")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    token=self.token,
                    trust_remote_code=True,
                )
                device_info = {"device": "cuda", "quantization": "4-bit"}
            except Exception as e:
                print(f"4-bit loading skipped ({e}). Falling back...")
                model = None

        if model is None and torch.cuda.is_available():
            try:
                print("CUDA available: Loading base model with float16...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    token=self.token,
                    trust_remote_code=True,
                )
                device_info = {"device": "cuda", "quantization": "fp16"}
            except Exception as e:
                print(f"CUDA float16 loading failed ({e}). Falling back to CPU...")
                model = None

        if model is None:
            print("Loading base model on CPU with float32...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=torch.float32,
                    device_map="cpu",
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=torch.float32,
                    device_map="cpu",
                    token=self.token,
                    trust_remote_code=True,
                )
            device_info = {"device": "cpu", "quantization": "fp32"}

        print(f"Loading LoRA adapter '{self.adapter_model_id}'...")
        attached = False
        load_error = None

        # Try 1: Load directly via token / Hub
        if self.token:
            try:
                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_model_id,
                    token=self.token,
                )
                attached = True
            except Exception as e:
                load_error = e

        # Try 2: If no token or Hub failed, check local cached snapshot
        if not attached:
            snapshot_path = resolve_local_adapter_snapshot(self.adapter_model_id)
            if snapshot_path:
                try:
                    model = PeftModel.from_pretrained(model, snapshot_path)
                    attached = True
                    print(f"Attached LoRA adapter from local snapshot cache: {snapshot_path}")
                except Exception as e:
                    load_error = e

        # Try 3: Standard Hub call without token (in case repo became public)
        if not attached:
            try:
                model = PeftModel.from_pretrained(model, self.adapter_model_id)
                attached = True
            except Exception as e:
                load_error = e

        if attached:
            self.adapter_loaded = True
            print("LoRA adapter merged/attached successfully.")
        else:
            if not self.token:
                auth_msg = (
                    f"Hugging Face repository '{self.adapter_model_id}' is private or requires authentication. "
                    "Please set the 'HF_TOKEN' environment variable (e.g., $env:HF_TOKEN = 'your_token' "
                    "or in a .env file) with read access."
                )
            else:
                auth_msg = f"Failed to load LoRA adapter '{self.adapter_model_id}': {load_error}"

            if self.allow_base_fallback:
                print(f"Notice: {auth_msg}\nProceeding with base model for smoke testing.")
                self.adapter_loaded = False
            else:
                raise RuntimeError(auth_msg)

        model.eval()
        return tokenizer, model, device_info

    @classmethod
    def get_instance(
        cls,
        force_reload: bool = False,
        allow_base_fallback: bool = False,
    ) -> "CricketTalkModel":
        """Get or initialize the cached singleton model instance."""
        if cls._instance is None or force_reload:
            cls._instance = cls(allow_base_fallback=allow_base_fallback)
        return cls._instance


def load_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_model_id: str = ADAPTER_MODEL_ID,
    force_reload: bool = False,
    allow_base_fallback: bool = False,
) -> CricketTalkModel:
    """Load and cache the model and tokenizer once."""
    if CricketTalkModel._instance is None or force_reload:
        CricketTalkModel._instance = CricketTalkModel(
            base_model_id=base_model_id,
            adapter_model_id=adapter_model_id,
            allow_base_fallback=allow_base_fallback,
        )
    return CricketTalkModel._instance


def generate_response(
    message: str,
    history: Optional[List[Union[Dict[str, str], List[str], Tuple[str, str]]]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    model_wrapper: Optional[CricketTalkModel] = None,
) -> str:
    """Generate a response using the Qwen chat template and LoRA model.

    Args:
        message: The user's input question.
        history: Conversation history (either list of dicts or list of [user, bot] pairs).
        system_prompt: System prompt instructing the commentary persona.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Penalty for repeated tokens.
        model_wrapper: Pre-loaded CricketTalkModel instance (defaults to singleton).

    Returns:
        Generated answer string.
    """
    if not message or not str(message).strip():
        return "Please enter a question or topic to get started!"

    wrapper = model_wrapper or CricketTalkModel.get_instance()
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for turn in history:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": turn["role"], "content": str(turn["content"])})
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_content, bot_content = turn
                if user_content:
                    messages.append({"role": "user", "content": str(user_content)})
                if bot_content:
                    messages.append({"role": "assistant", "content": str(bot_content)})

    messages.append({"role": "user", "content": str(message).strip()})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off prompt tokens to extract only the generated response
    prompt_len = inputs.input_ids.shape[1]
    response_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    return response
