# -*- coding: utf-8 -*-
"""
SST-2 sentiment classification via LLM prompting (Groq / Llama).
Strategies: zero-shot, one-shot, few-shot on 20 fixed validation examples.
"""

import os
import re
import random

from datasets import load_dataset
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEED = 42
random.seed(SEED)

ONE_SHOT_EXAMPLES = 1
FEW_SHOT_EXAMPLES = 3
TEST_SUBSET = 20
MAX_TEXT_LEN = 200

OUTPUT_FORMAT = (
    f"Output exactly {TEST_SUBSET} lines, one per test index, in this format:\n"
    "0. 0\n1. 1\n...\n"
    f"Use indices 0 to {TEST_SUBSET - 1} and label 0 (negative) or 1 (positive). "
    "Do not add explanations."
)


def ask_llama(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return chat.choices[0].message.content


def get_labels(block: str) -> list[int]:
    """Parse [0, 1, 0] or [0 1 0] style lists."""
    inner = block.strip().strip("[]")
    labels = []
    for part in re.split(r"[,;\s]+", inner):
        part = part.strip()
        if part in ("0", "1"):
            labels.append(int(part))
    return labels


def parse_predictions(response: str, n: int) -> tuple[list[int] | None, str | None]:
    """
    Extract binary labels (0/1) from the model response.
    Returns (labels, warning) where warning is set if fewer than n labels were found.
    """
    warning = None

    # 1. Numbered lines: "1. 1", "2: 0", "10. 1"
    numbered = re.findall(
        r"^\s*(\d+)\s*[\.\):\-]\s*([01])\s*$",
        response,
        re.MULTILINE,
    )
    if numbered:
        numbered.sort(key=lambda x: int(x[0]))
        labels = [int(label) for _, label in numbered]
        if len(labels) >= n:
            return labels[:n], None
        if labels:
            warning = f"only {len(labels)}/{n} labels (numbered list)"
            return labels, warning

    # 2. One label per line: [0] or [1] on its own line
    per_line = re.findall(r"^\s*\[([01])\]\s*$", response, re.MULTILINE)
    if len(per_line) >= n:
        return [int(x) for x in per_line[:n]], None
    if len(per_line) >= 1:
        warning = f"only {len(per_line)}/{n} labels (one per line)"
        labels = [int(x) for x in per_line]
        if len(labels) >= n:
            return labels[:n], None
        return labels, warning

    # 3. Single comma-separated bracket list: [0, 1, 0, ...]
    for block in re.findall(r"\[[^\]]+\]", response):
        labels = get_labels(block)
        if len(labels) >= 2:
            if len(labels) >= n:
                return labels[:n], None
            warning = f"only {len(labels)}/{n} labels (bracket list)"
            return labels, warning

    # 4. Fallback: scan all 0/1 tokens in order (works for many formats)
    digits = re.findall(r"\b[01]\b", response)
    if len(digits) >= n:
        return [int(d) for d in digits[:n]], None
    if digits:
        return [int(d) for d in digits], f"only {len(digits)}/{n} labels (token scan)"

    return None, None


def build_test_block(test_data) -> str:
    block = ""
    for i in range(TEST_SUBSET):
        text = test_data[i]["sentence"][:MAX_TEXT_LEN]
        block += f"[{i}]{text}"
    return block


def build_examples_block(train_data, n_examples: int) -> str:
    block = ""
    for i in range(n_examples):
        text = train_data[i]["sentence"][:MAX_TEXT_LEN]
        label = train_data[i]["label"]
        block += f"R-{text} [{label}].\n------------\n"
    return block


def run_strategy(name: str, prompt: str, gold_labels: list[int]) -> dict:
    print(f"{'=' * 55}")
    print(f" {name}")
    print(f"{'=' * 55}\n")

    try:
        response = ask_llama(prompt)
    except Exception as e:
        print(f"Error: {e}\n")
        return {"name": name, "error": str(e), "accuracy": None}

    print("Model response:")
    print(response)
    print()

    predicted, parse_warning = parse_predictions(response, TEST_SUBSET)
    print(f"Gold labels:      {gold_labels}")

    if predicted is None:
        print("Parsed labels:    (failed to parse)")
        accuracy = None
    else:
        print(f"Parsed labels:    {predicted}")
        if parse_warning:
            print(f"Parse note:       {parse_warning}")
        n_eval = min(len(predicted), len(gold_labels))
        correct = sum(predicted[i] == gold_labels[i] for i in range(n_eval))
        accuracy = correct / len(gold_labels)
        print(f"Accuracy:         {correct}/{len(gold_labels)} = {accuracy:.1%}")

    print()
    return {
        "name": name,
        "response": response,
        "predicted": predicted,
        "gold": gold_labels,
        "accuracy": accuracy,
    }


def zero_shot(test_data, gold_labels: list[int]) -> dict:
    base = (
        "We will provide a set of movie-review sentences from the SST-2 dataset. "
        "Classify each as positive or negative."
    )
    instruction = (
        "Each sentence is prefixed with an integer index in brackets, e.g. [0], followed by the text."
    )
    prompt = base + " " + instruction + " " + OUTPUT_FORMAT + build_test_block(test_data)
    return run_strategy("Zero-shot", prompt, gold_labels)


def one_shot(train_data, test_data, gold_labels: list[int]) -> dict:
    base = (
        "I will provide one example sentence with its sentiment label: "
        "0 = negative, 1 = positive. Each example starts with R- and ends with [0] or [1]."
    )
    instruction1 = (
        "After the example, a line with -------- separates examples from the sentences to classify."
    )
    prompt = (
        base + " " + instruction1 + " " + OUTPUT_FORMAT
        + build_examples_block(train_data, ONE_SHOT_EXAMPLES)
        + build_test_block(test_data)
    )
    return run_strategy("One-shot", prompt, gold_labels)


def few_shot(train_data, test_data, gold_labels: list[int]) -> dict:
    base = (
        "I will provide example sentences with sentiment labels: "
        "0 = negative, 1 = positive. Each example starts with R- and ends with [0] or [1]."
    )
    instruction1 = (
        "After the examples, a line with -------- separates examples from the sentences to classify."
    )
    prompt = (
        base + " " + instruction1 + " " + OUTPUT_FORMAT
        + build_examples_block(train_data, FEW_SHOT_EXAMPLES)
        + build_test_block(test_data)
    )
    return run_strategy(f"Few-shot ({FEW_SHOT_EXAMPLES} examples)", prompt, gold_labels)


def print_summary(results: list[dict]) -> None:
    print("=" * 55)
    print(" SUMMARY (20 shared validation examples)")
    print("=" * 55)
    for r in results:
        acc = r.get("accuracy")
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        print(f"  {r['name']:30s}  accuracy: {acc_str}")


if __name__ == "__main__":
    print("Loading SST-2 (GLUE)...")
    dataset = load_dataset("glue", "sst2")

    train_pool = dataset["train"].shuffle(seed=SEED)
    test_data = dataset["validation"].shuffle(seed=SEED).select(range(TEST_SUBSET))

    # Examples for one/few-shot come from train (disjoint from the 20 test sentences)
    train_for_shots = train_pool.select(range(FEW_SHOT_EXAMPLES))

    gold_labels = [test_data[i]["label"] for i in range(TEST_SUBSET)]

    print(f"Test examples: {TEST_SUBSET} (validation split, seed={SEED})")
    print(f"Gold labels:   {gold_labels}\n")

    if GROQ_API_KEY in ("", "....."):
        print(
            "Set GROQ_API_KEY (environment variable or edit this file) to call the API.\n"
            "Example: export GROQ_API_KEY='your-key-here'\n"
        )

    results = [
        zero_shot(test_data, gold_labels),
        one_shot(train_for_shots, test_data, gold_labels),
        few_shot(train_for_shots, test_data, gold_labels),
    ]
    print_summary(results)
