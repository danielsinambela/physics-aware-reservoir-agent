"""Ask the local gpt-oss:20b model (via Ollama) a Darcy-flow question.

Requires the Ollama app (or `ollama serve`) running locally and the model pulled:
    ollama pull gpt-oss:20b
Run:  python examples/test_local_llm.py

Only the local Ollama server is used; no API keys or external providers.
Set OLLAMA_HOST to use a non-default local address (default http://127.0.0.1:11434).
"""

import os
import sys
import time

import ollama

MODEL = "gpt-oss:20b"

SYSTEM_PROMPT = (
    "You are a scientific computing assistant specialising in partial differential "
    "equations, Darcy flow in porous media, and reservoir engineering. Give precise, "
    "mathematically rigorous answers, state assumptions, and use standard notation."
)

QUESTION = (
    "Explain the mathematical significance of harmonic permeability averaging in a "
    "conservative discretization of Darcy flow."
)


def main() -> int:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    client = ollama.Client(host=host, timeout=600)

    try:
        available = {m.model for m in client.list().models}
    except ConnectionError as exc:
        print(f"Cannot reach Ollama at {host}: {exc}\n"
              "Start the Ollama app or run `ollama serve`, then try again.", file=sys.stderr)
        return 1
    if MODEL not in available:
        print(f"Model {MODEL} is not installed. Run: ollama pull {MODEL}", file=sys.stderr)
        return 1

    print(f"Model: {MODEL} @ {host}\nQuestion: {QUESTION}\n")
    start = time.perf_counter()
    try:
        response = client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": QUESTION},
            ],
            options={"temperature": 0.2},
        )
    except ollama.ResponseError as exc:
        print(f"Ollama returned an error: {exc.error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    answer = (response.message.content or "").strip()
    if not answer:
        print("The model returned an empty response.", file=sys.stderr)
        return 1
    print(answer)
    print(f"\n[{response.eval_count or 0} tokens in {elapsed:.1f} s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
