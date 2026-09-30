from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    RequestSpec,
    make_approximate_token_prompt,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "02-chunked-prefill"

RESULTS_DIRECTORY = Path(
    "results/02-chunked-prefill"
)


def build_long_prefill_prompt() -> str:
    """
    Build a deliberately long prompt.

    The local helper is approximate because exact token count depends
    on the Qwen tokenizer. On MI300X, the scheduler trace will tell us
    exactly how many prompt tokens vLLM schedules.
    """

    long_context = make_approximate_token_prompt(
        approximate_token_count=3000,
        repeated_text=(
            "GPU memory inference scheduler attention "
            "request token cache execution "
        ),
    )

    return (
        "Read the following technical notes and summarize the main idea.\n\n"
        + long_context
    )


def build_requests() -> list[RequestSpec]:
    """
    R1 starts first and remains in decode.

    R2 arrives later with a large prompt.

    This creates the condition we care about:

        existing decode work
                +
        newly arriving long prefill
    """

    return [
        RequestSpec(
            request_id="R1-DECODE",
            prompt=(
                "Explain how a GPU executes numerical workloads. "
                "Continue with a detailed explanation of compute units, "
                "memory movement, scheduling, and inference."
            ),
            max_output_tokens=256,
            start_delay_seconds=0.00,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R2-LONG-PREFILL",
            prompt=build_long_prefill_prompt(),
            max_output_tokens=32,
            start_delay_seconds=0.10,
            temperature=0.0,
            ignore_eos=True,
        ),
    ]


def get_requests_for_mode(
    experiment_mode: str,
) -> list[RequestSpec]:
    """
    Client workload stays identical.

    Only the server scheduler configuration changes between modes.
    """

    if experiment_mode in {
        "baseline",
        "chunked",
    }:
        return build_requests()

    raise ValueError(
        f"Unsupported experiment mode: {experiment_mode}"
    )


def print_experiment_prediction(
    experiment_mode: str,
) -> None:

    print()
    print("Prediction")
    print("----------")

    if experiment_mode == "baseline":
        print(
            "R2's long prompt should be scheduled with the normal "
            "server token budget. We record how many scheduler "
            "iterations its prefill requires."
        )

    elif experiment_mode == "chunked":
        print(
            "With chunked prefill enabled and a deliberately small "
            "per-iteration token budget, R2's prompt should appear "
            "across multiple scheduler iterations while R1 continues "
            "to receive decode work."
        )

    print()


def print_server_requirement(
    experiment_mode: str,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    if experiment_mode == "baseline":
        print(
            "Normal Qwen3 server with TraceScheduler. "
            "Do not deliberately reduce max-num-batched-tokens."
        )

    elif experiment_mode == "chunked":
        print(
            "Same Qwen3 server with TraceScheduler, plus:"
        )
        print()
        print("  --enable-chunked-prefill")
        print("  --max-num-batched-tokens 512")

    print()


def save_results(
    experiment_mode: str,
    request_results,
    event_records,
) -> None:

    mode_results_directory = (
        RESULTS_DIRECTORY / experiment_mode
    )

    write_csv(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.csv"
        ),
    )

    write_csv(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.csv"
        ),
    )

    write_jsonl(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.jsonl"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.jsonl"
        ),
    )


def print_request_summary(
    request_results,
) -> None:

    print()
    print("Client-side request summary")
    print("---------------------------")

    print(
        f"{'Request':<22}"
        f"{'Start(s)':<12}"
        f"{'TTFT(s)':<12}"
        f"{'Total(s)':<12}"
    )

    print("-" * 58)

    for request_result in request_results:

        if request_result.time_to_first_token_seconds is None:
            time_to_first_token = "N/A"
        else:
            time_to_first_token = (
                f"{request_result.time_to_first_token_seconds:.4f}"
            )

        print(
            f"{request_result.request_id:<22}"
            f"{request_result.request_started_seconds:<12.4f}"
            f"{time_to_first_token:<12}"
            f"{request_result.total_request_seconds:<12.4f}"
        )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-02: Observe how vLLM divides a long prefill "
            "across scheduler iterations."
        )
    )

    argument_parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "chunked",
        ],
        required=True,
        help=(
            "baseline = normal scheduler token budget; "
            "chunked = small token budget forces long prefill chunks"
        ),
    )

    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned workload without contacting vLLM."
        ),
    )

    argument_parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the vLLM server.",
    )

    return argument_parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    request_specs = get_requests_for_mode(
        experiment_mode=arguments.mode,
    )

    print()
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Mode       : {arguments.mode}")

    print_experiment_prediction(
        experiment_mode=arguments.mode,
    )

    print_server_requirement(
        experiment_mode=arguments.mode,
    )

    if arguments.dry_run:
        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.mode,
        )
        return

    request_results, event_records = run_staggered_requests(
        request_specs=request_specs,
        experiment_name=EXPERIMENT_NAME,
        experiment_mode=arguments.mode,
        vllm_url=arguments.vllm_url,
    )

    save_results(
        experiment_mode=arguments.mode,
        request_results=request_results,
        event_records=event_records,
    )

    print_request_summary(
        request_results=request_results,
    )

    print(
        "Client evidence saved under:"
    )

    print(
        RESULTS_DIRECTORY
        / arguments.mode
    )

    print()
    print(
        "Primary mechanism evidence comes from [BATCH_TRACE]. "
        "Look for R2-LONG-PREFILL receiving hundreds of tokens "
        "across multiple scheduler steps while R1-DECODE continues "
        "receiving decode work."
    )
    print()


if __name__ == "__main__":
    main()
