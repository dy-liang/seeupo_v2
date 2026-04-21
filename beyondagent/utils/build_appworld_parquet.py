#!/usr/bin/env python3
"""Reconstruct AppWorld parquet datasets from split txt files.

This script rebuilds the parquet structure consumed by this project from
`appworld_data/data/datasets/*.txt`. The original preprocessing script is not
present in this repo, so this implementation reconstructs the same core schema
used by the existing `external/appworld/data/dev.parquet`:

- data_source: "appworld"
- prompt: chat messages compatible with the RL dataset loader
- reward_model: object column with a ground_truth placeholder
- extras: task metadata used later by rollout/trainer code

It reuses the AppWorld prompt template defined in
`env_service/environments/appworld/appworld_env.py` and fills in the task
supervisor information from each task's `specs.json`.

Usage: python build_appworld_parquet.py appworld_data/data/datasets/test_normal.txt  --output external/appworld/data/test_normal.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TXT_PATH = REPO_ROOT / "appworld_data" / "data" / "datasets" / "dev.txt"
DEFAULT_TASKS_ROOT = REPO_ROOT / "appworld_data" / "data" / "tasks"
DEFAULT_APPWORLD_ENV_PATH = (
    REPO_ROOT / "env_service" / "environments" / "appworld" / "appworld_env.py"
)
DEFAULT_API_DOCS_ROOT = REPO_ROOT / "appworld_data" / "data" / "api_docs" / "standard"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "external" / "appworld" / "data"
DEFAULT_TEMPLATE_PARQUET = DEFAULT_OUTPUT_DIR / "dev.parquet"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an AppWorld parquet dataset from a split txt file."
    )
    parser.add_argument(
        "txt_path",
        nargs="?",
        default=str(DEFAULT_TXT_PATH),
        help=f"Path to the split txt file. Default: {DEFAULT_TXT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Default: external/appworld/data/<split>.parquet",
    )
    parser.add_argument(
        "--jsonl-output",
        default=None,
        help="Optional JSONL dump of the reconstructed rows for debugging.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=2,
        help="Number of preview rows to print. Default: 2",
    )
    parser.add_argument(
        "--tasks-root",
        default=str(DEFAULT_TASKS_ROOT),
        help=f"Task directory root. Default: {DEFAULT_TASKS_ROOT}",
    )
    parser.add_argument(
        "--api-docs-root",
        default=str(DEFAULT_API_DOCS_ROOT),
        help=f"API docs root used to enumerate available apps. Default: {DEFAULT_API_DOCS_ROOT}",
    )
    parser.add_argument(
        "--appworld-env-path",
        default=str(DEFAULT_APPWORLD_ENV_PATH),
        help=f"Path to appworld_env.py used to extract the prompt template. Default: {DEFAULT_APPWORLD_ENV_PATH}",
    )
    parser.add_argument(
        "--template-parquet",
        default=str(DEFAULT_TEMPLATE_PARQUET),
        help="Existing parquet used as the canonical prompt template source.",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Only print preview and optional JSONL; do not attempt parquet writing.",
    )
    return parser.parse_args()


def read_task_ids(txt_path: Path) -> list[str]:
    task_ids = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not task_ids:
        raise ValueError(f"No task ids found in {txt_path}")
    return task_ids


def infer_split_name(txt_path: Path) -> str:
    return txt_path.stem


def infer_output_path(txt_path: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    split = infer_split_name(txt_path)
    return (DEFAULT_OUTPUT_DIR / f"{split}.parquet").resolve()


def extract_prompt_template(appworld_env_path: Path) -> str:
    source = appworld_env_path.read_text(encoding="utf-8")
    match = re.search(r'simple_prompt\s*=\s*"""(.*?)"""', source, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find simple_prompt in {appworld_env_path}")
    return match.group(1)


def render_prompt_template(template: str, supervisor: dict[str, Any]) -> str:
    rendered = template
    replacements = {
        "{{ main_user.first_name }}": str(supervisor["first_name"]),
        "{{ main_user.last_name }}": str(supervisor["last_name"]),
        "{{ main_user.email }}": str(supervisor["email"]),
        "{{ main_user.phone_number }}": str(supervisor["phone_number"]),
        "{{ supervisor.first_name }}": str(supervisor["first_name"]),
        "{{ supervisor.last_name }}": str(supervisor["last_name"]),
        "{{ supervisor.email }}": str(supervisor["email"]),
        "{{ supervisor.phone_number }}": str(supervisor["phone_number"]),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def parse_prompt_messages(rendered_prompt: str, instruction: str) -> list[dict[str, str]]:
    chunks = re.split(r"(?m)^(USER|ASSISTANT):\s*$", rendered_prompt.strip())
    if len(chunks) < 3:
        raise ValueError("Rendered prompt does not contain USER/ASSISTANT blocks as expected.")

    messages: list[dict[str, str]] = []
    i = 1
    while i < len(chunks):
        role_token = chunks[i]
        content = chunks[i + 1].strip()
        role = "user" if role_token == "USER" else "assistant"
        if content:
            messages.append({"role": role, "content": content})
        i += 2

    if not messages:
        raise ValueError("No messages parsed from rendered prompt.")

    last_content = messages[-1]["content"].rstrip()
    if last_content.endswith("Task:"):
        messages[-1]["content"] = f"{last_content}\n\n{instruction}"
    else:
        messages[-1]["content"] = f"{last_content}\n\nTask:\n\n{instruction}"
    return messages


def load_prompt_template_from_parquet(template_parquet: Path) -> list[dict[str, str]] | None:
    if not template_parquet.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None
    rows = pq.read_table(template_parquet).slice(0, 1).to_pylist()
    if not rows:
        return None
    prompt = rows[0].get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 15:
        return None
    return prompt


def load_schema_metadata_from_parquet(template_parquet: Path) -> dict[bytes, bytes] | None:
    if not template_parquet.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None
    pf = pq.ParquetFile(template_parquet)
    return pf.schema_arrow.metadata


def substitute_supervisor_line(text: str, supervisor: dict[str, Any]) -> str:
    replacement = (
        f"My name is: {supervisor['first_name']} {supervisor['last_name']}. "
        f"My personal email is {supervisor['email']} and phone number is {supervisor['phone_number']}."
    )
    return re.sub(
        r"My name is: .*? My personal email is .*? and phone number is .*?\.",
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )


def substitute_instruction_block(text: str, instruction: str) -> str:
    return re.sub(r"(Task:\n\n)(.*)$", rf"\1{instruction}", text, count=1, flags=re.DOTALL)


def build_prompt_from_parquet_template(
    template_prompt: list[dict[str, str]],
    supervisor: dict[str, Any],
    instruction: str,
) -> list[dict[str, str]]:
    prompt = json.loads(json.dumps(template_prompt))
    prompt[0]["content"] = substitute_supervisor_line(prompt[0]["content"], supervisor)
    prompt[-1]["content"] = substitute_supervisor_line(prompt[-1]["content"], supervisor)
    prompt[-1]["content"] = substitute_instruction_block(prompt[-1]["content"], instruction)
    return prompt


def load_specs(task_root: Path) -> dict[str, Any]:
    specs_path = task_root / "specs.json"
    if not specs_path.exists():
        raise FileNotFoundError(f"Missing specs.json for task: {task_root.name}")
    return json.loads(specs_path.read_text(encoding="utf-8"))


def load_ground_truth_answer(task_root: Path) -> str | None:
    answer_path = task_root / "ground_truth" / "answer.json"
    if not answer_path.exists():
        return None
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    if answer is None:
        return "None"
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def build_row(
    task_id: str,
    index: int,
    split: str,
    tasks_root: Path,
    template: str,
    template_prompt: list[dict[str, str]] | None,
) -> dict[str, Any]:
    task_root = tasks_root / task_id
    specs = load_specs(task_root)
    supervisor = specs["supervisor"]
    if template_prompt is not None:
        prompt = build_prompt_from_parquet_template(template_prompt, supervisor, specs["instruction"])
    else:
        rendered_prompt = render_prompt_template(template, supervisor)
        prompt = parse_prompt_messages(rendered_prompt, specs["instruction"])
    ground_truth = load_ground_truth_answer(task_root)

    return {
        "data_source": "appworld",
        "prompt": prompt,
        "reward_model": {"ground_truth": ground_truth},
        "extras": {
            "index": index,
            "prompt_block_len": len(prompt),
            "split": split,
            "task_id": task_id,
        },
    }


def build_rows(
    txt_path: Path,
    tasks_root: Path,
    appworld_env_path: Path,
    template_parquet: Path,
) -> list[dict[str, Any]]:
    task_ids = read_task_ids(txt_path)
    split = infer_split_name(txt_path)
    template = extract_prompt_template(appworld_env_path)
    template_prompt = load_prompt_template_from_parquet(template_parquet)

    rows = []
    for index, task_id in enumerate(task_ids):
        rows.append(
            build_row(
                task_id=task_id,
                index=index,
                split=split,
                tasks_root=tasks_root,
                template=template,
                template_prompt=template_prompt,
            )
        )
    return rows


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_write_with_pandas(rows: list[dict[str, Any]], output_path: Path) -> str | None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    ensure_parent_dir(output_path)
    df = pd.DataFrame(rows, columns=["data_source", "prompt", "reward_model", "extras"])
    df.to_parquet(output_path, index=None, compression="snappy")
    return "pandas"


def try_write_with_pyarrow(
    rows: list[dict[str, Any]],
    output_path: Path,
    schema_metadata: dict[bytes, bytes] | None = None,
) -> str | None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    ensure_parent_dir(output_path)
    table = pa.Table.from_pylist(rows)
    if schema_metadata is not None:
        table = table.replace_schema_metadata(schema_metadata)
    pq.write_table(table, output_path, compression="snappy")
    return "pyarrow"


def write_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    schema_metadata: dict[bytes, bytes] | None = None,
) -> str:
    backend = try_write_with_pyarrow(rows, output_path, schema_metadata=schema_metadata)
    if backend is not None:
        return backend

    backend = try_write_with_pandas(rows, output_path)
    if backend is not None:
        return backend
    raise ModuleNotFoundError(
        "No parquet writer backend found. Install `pyarrow` or `pandas` in the target environment."
    )


def shorten_json(value: Any, max_len: int = 260) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_preview(rows: list[dict[str, Any]], preview_rows: int) -> None:
    print(f"rows: {len(rows)}")
    print("columns: ['data_source', 'prompt', 'reward_model', 'extras']")
    preview_rows = max(0, min(preview_rows, len(rows)))
    for idx in range(preview_rows):
        row = rows[idx]
        print(f"\n[row {idx}]")
        preview = {
            "data_source": row["data_source"],
            "prompt": shorten_json(row["prompt"]),
            "reward_model": shorten_json(row["reward_model"]),
            "extras": shorten_json(row["extras"]),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    txt_path = Path(args.txt_path).expanduser().resolve()
    tasks_root = Path(args.tasks_root).expanduser().resolve()
    api_docs_root = Path(args.api_docs_root).expanduser().resolve()
    appworld_env_path = Path(args.appworld_env_path).expanduser().resolve()
    template_parquet = Path(args.template_parquet).expanduser().resolve()

    if not txt_path.exists():
        print(f"Split txt not found: {txt_path}", file=sys.stderr)
        return 1
    if not tasks_root.exists():
        print(f"Tasks root not found: {tasks_root}", file=sys.stderr)
        return 1
    if not api_docs_root.exists():
        print(f"API docs root not found: {api_docs_root}", file=sys.stderr)
        return 1
    if not appworld_env_path.exists():
        print(f"appworld_env.py not found: {appworld_env_path}", file=sys.stderr)
        return 1

    rows = build_rows(
        txt_path=txt_path,
        tasks_root=tasks_root,
        appworld_env_path=appworld_env_path,
        template_parquet=template_parquet,
    )
    schema_metadata = load_schema_metadata_from_parquet(template_parquet)

    print_preview(rows, args.preview_rows)

    if args.jsonl_output:
        jsonl_output = Path(args.jsonl_output).expanduser().resolve()
        write_jsonl(rows, jsonl_output)
        print(f"\njsonl_written_to: {jsonl_output}")

    if args.skip_parquet:
        return 0

    output_path = infer_output_path(txt_path, args.output)
    try:
        backend = write_parquet(rows, output_path, schema_metadata=schema_metadata)
    except ModuleNotFoundError as exc:
        print(f"\nparquet_write_failed: {exc}", file=sys.stderr)
        print(
            "Tip: rerun in the training environment or install `pyarrow`/`pandas`, "
            "or use --jsonl-output first to inspect reconstructed rows.",
            file=sys.stderr,
        )
        return 2

    print(f"\nparquet_written_to: {output_path}")
    print(f"parquet_backend: {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
