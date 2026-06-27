from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct"

DEFAULT_DOCUMENT = """
vLLM is an LLM serving engine designed for high-throughput inference.
It uses techniques such as paged KV cache management and continuous batching
to improve serving efficiency.

In a document question answering system, the business request starts from a
user document and a user question. The system retrieves relevant context,
builds a prompt, sends it to a local LLM, and returns the generated answer.

The first baseline should be simple and measurable. Later phases will compare
TTFT, TPOT, tokens per second, peak GPU memory, and concurrency behavior.
"""

DEFAULT_QUESTION = "Which techniques does the document say vLLM uses to improve serving efficiency?"


@dataclass
class RagResult:
    document: str
    question: str
    retrieved_context: str
    prompt: str
    answer: str


class LocalGenerator:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            device_map=self.device,
            local_files_only=True,
        )

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful document question answering assistant."},
            {"role": "user", "content": prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer_ids = generated_ids[0][model_inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_generator(model_path_text: str = str(DEFAULT_MODEL_PATH)) -> LocalGenerator:
    return LocalGenerator(Path(model_path_text))


def split_document(document: str, max_chars: int = 420) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", document) if part.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        for start in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[start : start + max_chars].strip())

    return chunks


def retrieve_context(document: str, question: str, top_k: int = 2) -> str:
    chunks = split_document(document)
    question_terms = set(_tokenize(question))

    scored_chunks = []
    for index, chunk in enumerate(chunks):
        chunk_terms = set(_tokenize(chunk))
        overlap_score = len(question_terms & chunk_terms)
        scored_chunks.append((overlap_score, index, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    selected_chunks = [chunk for _, _, chunk in scored_chunks[:top_k]]
    return "\n\n".join(selected_chunks)


def build_prompt(retrieved_context: str, question: str) -> str:
    return f"""You are a document question answering assistant.
Answer only from the document context.
Keep the answer short.
Do not add explanations that are not explicitly present in the document context.
If the context is insufficient, say you do not know.

Document context:
{retrieved_context}

Question:
{question}

Answer:
"""


def generate_answer(
    prompt: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    max_new_tokens: int = 128,
) -> str:
    return get_generator(str(model_path)).generate(prompt, max_new_tokens=max_new_tokens)


def run_rag(document: str, question: str, model_path: Path = DEFAULT_MODEL_PATH) -> RagResult:
    retrieved_context = retrieve_context(document, question)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt, model_path=model_path)

    return RagResult(
        document=document.strip(),
        question=question,
        retrieved_context=retrieved_context,
        prompt=prompt,
        answer=answer,
    )


def _tokenize(text: str) -> Iterable[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
