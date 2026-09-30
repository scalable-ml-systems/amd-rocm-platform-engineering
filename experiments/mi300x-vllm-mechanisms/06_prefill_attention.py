from __future__ import annotations

import argparse
import csv
import json
import urllib.request

from pathlib import Path

from common import (
    DEFAULT_MODEL_NAME,
    RequestSpec,
    make_approximate_token_prompt,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "06-prefill-attention"

RESULTS_DIRECTORY = Path(
    "results/06-prefill-attention"
)

TARGET_PROMPT_SIZES = [
    256,
    1024,
    4096,
]


def build_prompt(
    approximate_token_count: int,
) -> str:
    """
    Construct a deterministic prompt large enough to exercise
    prefill attention.

    Exact token count will later be measured with the running
    Qwen tokenizer.
    """

    technical_context = make_approximate_token_prompt(
        approximate_token_count=approximate_token_count,
        repeated_text=(
            "GPU attention memory query key value "
            "prefill inference compute bandwidth tile "
        ),
    )

    return (
        "Read the technical context below and answer with one word.\n\n"
        + technical_context
        + "\n\nQuestion: What hardware executes this workload?"
    )


def build_requests() -> list[RequestSpec]:
    """
    Each request generates only one output token.

    That keeps the experiment dominated by prompt prefill instead
    of autoregressive decode.
    """

    request_specs = []

    for approximate_token_count in TARGET_PROMPT_SIZES:

        request_specs.append(
            RequestSpec(
                request_id=(
                    f"PREFILL-{approximate_token_count}"
                ),
                prompt=build_prompt(
                    approximate_token_count
                ),
                max_output_tokens=1,
                start_delay_seconds=0.0,
                temperature=0.0,
                ignore_eos=True,
            )
        )

    return request_specs


def get_exact_prompt_token_count(
    prompt: str,
    vllm_url: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """
    Ask vLLM to tokenize the actual chat request.

    This avoids pretending that words or characters equal model tokens.
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

    return int(
        tokenize_response["count"]
    )


def write_prefill_summary(
    request_specs: list[RequestSpec],
    request_results,
    vllm_url: str,
    output_path: Path,
) -> None:
    """
    Record exact prompt length alongside client-observed latency.

    TTFT is useful here because the request generates only one token,
    so most of the request's work occurs during prefill.
    """

    result_by_request_id = {
        result.request_id: result
        for result in request_results
    }

    summary_rows = []

    for request_spec in request_specs:

        exact_prompt_tokens = (
            get_exact_prompt_token_count(
                prompt=request_spec.prompt,
                vllm_url=vllm_url,
            )
        )

        request_result = (
            result_by_request_id[
                request_spec.request_id
            ]
        )

        summary_rows.append(
            {
                "request_id":
                    request_spec.request_id,

                "exact_prompt_tokens":
                    exact_prompt_tokens,

                "time_to_first_token_seconds":
                    request_result.time_to_first_token_seconds,

                "total_request_seconds":
                    request_result.total_request_seconds,
            }
        )

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
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def print_prediction(
    experiment_mode: str,
) -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        "As prompt length increases, prefill attention performs "
        "substantially more attention work."
    )

    print(
        "The optimized ROCm attention backend should tile the "
        "attention calculation instead of materializing the entire "
        "attention-score matrix in HBM."
    )

    if experiment_mode == "default":
        print(
            "We first observe whichever attention backend vLLM "
            "selects by default in the current ROCm image."
        )

    elif experiment_mode == "aiter":
        print(
            "With AITER enabled, Qwen's MHA attention should use "
            "an AITER attention path when supported."
        )

    print()


def print_server_requirement(
    experiment_mode: str,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    if experiment_mode == "default":

        print(
            "Normal Qwen3 server. Do not force an attention backend."
        )

    elif experiment_mode == "aiter":

        print(
            "Same Qwen3 server with:"
        )

        print()
        print(
            "  VLLM_ROCM_USE_AITER=1"
        )

        print()

        print(
            "Do not force --attention-backend initially. "
            "Record which backend vLLM auto-selects."
        )

    print()


def print_evidence_goal() -> None:

    print("Evidence goal")
    print("-------------")

    print(
        "For each prompt length:"
    )

    print(
        "  exact prompt tokens"
    )

    print(
        "  selected ROCm attention backend"
    )

    print(
        "  actual attention kernel name"
    )

    print(
        "  prefill execution time"
    )

    print(
        "  targeted memory traffic if profiler capture is needed"
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-06: Observe how MI300X executes long-prompt "
            "prefill attention as sequence length grows."
        )
    )

    argument_parser.add_argument(
        "--mode",
        choices=[
            "default",
            "aiter",
        ],
        required=True,
        help=(
            "default = use normal vLLM backend selection; "
            "aiter = start server with VLLM_ROCM_USE_AITER=1"
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

    request_specs = build_requests()

    print()
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Mode       : {arguments.mode}"
    )

    print_prediction(
        experiment_mode=arguments.mode,
    )

    print_server_requirement(
        experiment_mode=arguments.mode,
    )

    print_evidence_goal()

    if arguments.dry_run:

        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.mode,
        )

        return

    request_results, event_records = (
        run_staggered_requests(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.mode,
            vllm_url=arguments.vllm_url,
        )
    )

    mode_results_directory = (
        RESULTS_DIRECTORY
        / arguments.mode
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

    write_prefill_summary(
        request_specs=request_specs,
        request_results=request_results,
        vllm_url=arguments.vllm_url,
        output_path=(
            mode_results_directory
            / "prefill-summary.csv"
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
        "Primary mechanism evidence still comes from the "
        "server startup log and targeted kernel trace:"
    )

    print(
        "prompt length -> attention backend -> "
        "attention kernel -> memory behavior"
    )

    print()


if __name__ == "__main__":
    main()
