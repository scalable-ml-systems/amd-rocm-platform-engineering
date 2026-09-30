from __future__ import annotations

import argparse
import csv
import json
import urllib.request

from pathlib import Path

from common import (
    DEFAULT_MODEL_NAME,
    RequestSpec,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "04-paged-attention"

RESULTS_DIRECTORY = Path(
    "results/04-paged-attention"
)

OUTPUT_TOKENS = 64


def build_request() -> RequestSpec:
    """
    One request is enough.

    The prompt establishes an existing KV cache.
    The 64 generated tokens then force the logical sequence
    to cross additional KV-cache block boundaries.
    """

    return RequestSpec(
        request_id="R1-PAGED-KV",
        prompt=(
            "Explain how a GPU memory hierarchy works, including "
            "registers, local memory, caches, and HBM. Then explain "
            "how an inference system can store growing attention state "
            "without requiring one contiguous memory allocation."
        ),
        max_output_tokens=OUTPUT_TOKENS,
        start_delay_seconds=0.0,
        temperature=0.0,
        ignore_eos=True,
    )


def tokenize_chat_prompt(
    prompt: str,
    vllm_url: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """
    Ask the running vLLM server for the exact Qwen chat-template
    token count.

    We do not guess prompt length locally because chat-template
    tokens also belong to the sequence stored in the KV cache.
    """

    tokenize_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "add_generation_prompt": True,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    encoded_payload = json.dumps(
        tokenize_payload
    ).encode("utf-8")

    tokenize_request = urllib.request.Request(
        url=f"{vllm_url}/tokenize",
        data=encoded_payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        tokenize_request,
        timeout=30,
    ) as response:
        tokenize_response = json.loads(
            response.read().decode("utf-8")
        )

    return int(tokenize_response["count"])


def calculate_logical_block_number(
    token_position: int,
    block_size: int,
) -> int:
    """
    Convert a zero-based token position into its logical KV block.

    Example with block_size=16:

        token 0..15   -> logical block 0
        token 16..31  -> logical block 1
        token 32..47  -> logical block 2
    """

    return token_position // block_size


def build_block_boundary_records(
    prompt_token_count: int,
    output_token_count: int,
    block_size: int,
) -> list[dict]:
    """
    Describe only the points where generation enters a new logical block.

    This is the logical view.

    The physical block IDs will come from vLLM's KV-cache manager
    during the GPU experiment.
    """

    records: list[dict] = []

    total_sequence_tokens = (
        prompt_token_count
        + output_token_count
    )

    previous_logical_block = None

    for token_position in range(
        prompt_token_count,
        total_sequence_tokens,
    ):
        logical_block = (
            calculate_logical_block_number(
                token_position=token_position,
                block_size=block_size,
            )
        )

        if logical_block == previous_logical_block:
            continue

        generated_token_number = (
            token_position
            - prompt_token_count
            + 1
        )

        records.append(
            {
                "sequence_token_position":
                    token_position,

                "generated_token_number":
                    generated_token_number,

                "logical_block":
                    logical_block,

                "block_size":
                    block_size,
            }
        )

        previous_logical_block = logical_block

    return records


def write_dictionary_csv(
    records: list[dict],
    output_path: Path,
) -> None:

    if not records:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)


def print_logical_block_plan(
    prompt_token_count: int,
    block_size: int,
) -> None:

    total_sequence_tokens = (
        prompt_token_count
        + OUTPUT_TOKENS
    )

    starting_logical_block = (
        calculate_logical_block_number(
            token_position=prompt_token_count - 1,
            block_size=block_size,
        )
    )

    ending_logical_block = (
        calculate_logical_block_number(
            token_position=total_sequence_tokens - 1,
            block_size=block_size,
        )
    )

    total_logical_blocks = (
        ending_logical_block + 1
    )

    print()
    print("Logical KV layout")
    print("-----------------")

    print(
        f"Prompt tokens       : {prompt_token_count}"
    )

    print(
        f"Generated tokens    : {OUTPUT_TOKENS}"
    )

    print(
        f"KV block size       : {block_size}"
    )

    print(
        f"Prompt ends in block: {starting_logical_block}"
    )

    print(
        f"Sequence ends block : {ending_logical_block}"
    )

    print(
        f"Logical blocks used : {total_logical_blocks}"
    )

    print()


def print_prediction(
    block_size: int,
) -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        f"With a KV block size of {block_size}, every "
        f"{block_size} logical sequence tokens require another "
        "KV-cache page."
    )

    print(
        "Those logical pages do NOT need to be physically adjacent. "
        "PagedAttention should use the request's block table to find "
        "the physical KV pages during decode."
    )

    print()


def print_server_requirement(
    block_size: int,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    print(
        "Qwen3 server with TraceScheduler and:"
    )

    print()
    print(
        f"  --block-size {block_size}"
    )

    print()

    print(
        "For the GPU execution phase, the scheduler tracer will also "
        "record the physical KV block IDs assigned to R1-PAGED-KV."
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-04: Connect logical token positions to paged "
            "KV-cache blocks and the real PagedAttention path."
        )
    )

    argument_parser.add_argument(
        "--block-size",
        type=int,
        choices=[
            16,
            32,
        ],
        required=True,
        help=(
            "ROCm PagedAttention KV-cache block size."
        ),
    )

    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    argument_parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
    )

    return argument_parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    request_spec = build_request()

    print()
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Block size : {arguments.block_size}")

    print_prediction(
        block_size=arguments.block_size,
    )

    print_server_requirement(
        block_size=arguments.block_size,
    )

    if arguments.dry_run:

        print_dry_run(
            request_specs=[request_spec],
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=(
                f"block-{arguments.block_size}"
            ),
        )

        print(
            "Exact logical block boundaries will be calculated "
            "on MI300X using vLLM's /tokenize endpoint."
        )

        return

    prompt_token_count = tokenize_chat_prompt(
        prompt=request_spec.prompt,
        vllm_url=arguments.vllm_url,
    )

    block_boundary_records = (
        build_block_boundary_records(
            prompt_token_count=prompt_token_count,
            output_token_count=OUTPUT_TOKENS,
            block_size=arguments.block_size,
        )
    )

    print_logical_block_plan(
        prompt_token_count=prompt_token_count,
        block_size=arguments.block_size,
    )

    experiment_mode = (
        f"block-{arguments.block_size}"
    )

    request_results, event_records = (
        run_staggered_requests(
            request_specs=[request_spec],
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=experiment_mode,
            vllm_url=arguments.vllm_url,
        )
    )

    mode_results_directory = (
        RESULTS_DIRECTORY
        / experiment_mode
    )

    write_csv(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.csv"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.jsonl"
        ),
    )

    write_dictionary_csv(
        records=block_boundary_records,
        output_path=(
            mode_results_directory
            / "logical-block-boundaries.csv"
        ),
    )

    print(
        "Evidence saved under:"
    )

    print(
        mode_results_directory
    )

    print()

    print(
        "Next evidence to correlate:"
    )

    print(
        "logical block -> physical KV block ID -> "
        "paged_attention kernel"
    )

    print()


if __name__ == "__main__":
    main()
