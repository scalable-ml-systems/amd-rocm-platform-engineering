from __future__ import annotations

import argparse
import csv
import time
import urllib.request

from pathlib import Path

from common import (
    RequestSpec,
    print_dry_run,
    send_streaming_request,
)


EXPERIMENT_NAME = "07-speculative-decoding"

RESULTS_DIRECTORY = Path(
    "results/07-speculative-decoding"
)


SPECULATIVE_METRIC_NAMES = {
    "verification_steps": {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_drafts_total",
    },
    "draft_tokens": {
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_draft_tokens_total",
    },
    "accepted_tokens": {
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_total",
    },
}


def build_predictable_request() -> RequestSpec:
    """
    Give the n-gram proposer an obvious sequence that already
    appears repeatedly in the prompt.

    Prediction:
    prompt lookup should often find useful continuations.
    """

    repeated_pattern = (
        "red blue green yellow "
        * 80
    )

    prompt = (
        "Study this repeating sequence:\n\n"
        f"{repeated_pattern}\n\n"
        "Continue the same sequence exactly. "
        "Do not explain it. Output only the sequence."
    )

    return RequestSpec(
        request_id="PREDICTABLE",
        prompt=prompt,
        max_output_tokens=128,
        temperature=0.0,
        ignore_eos=True,
    )


def build_unpredictable_request() -> RequestSpec:
    """
    Ask for novel text with little deliberate repetition.

    Prediction:
    prompt lookup should find fewer useful continuations,
    causing more proposed tokens to be rejected.

    This is a prediction, not an assumption. The measured
    acceptance counters decide whether it is true.
    """

    prompt = (
        "Describe an imaginary failure involving a GPU scheduler, "
        "network fabric, storage service, and inference router. "
        "Invent a different failure mechanism for each component "
        "and explain the causal chain without repeating phrases."
    )

    return RequestSpec(
        request_id="UNPREDICTABLE",
        prompt=prompt,
        max_output_tokens=128,
        temperature=0.0,
        ignore_eos=True,
    )


def fetch_speculative_metrics(
    metrics_url: str,
) -> dict[str, float]:
    """
    Read vLLM's aggregate speculative-decoding counters.

    We accept both the internal metric names and Prometheus
    '_total' counter names so the experiment is tolerant of
    exposition naming differences.
    """

    observed_metrics = {
        "verification_steps": 0.0,
        "draft_tokens": 0.0,
        "accepted_tokens": 0.0,
    }

    with urllib.request.urlopen(
        metrics_url,
        timeout=5,
    ) as response:
        metrics_text = response.read().decode("utf-8")

    for line in metrics_text.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        metric_fields = line.split()

        if len(metric_fields) < 2:
            continue

        metric_name = metric_fields[0].split("{", 1)[0]

        try:
            metric_value = float(metric_fields[1])
        except ValueError:
            continue

        for (
            logical_metric_name,
            accepted_metric_names,
        ) in SPECULATIVE_METRIC_NAMES.items():

            if metric_name in accepted_metric_names:
                observed_metrics[
                    logical_metric_name
                ] += metric_value

    return observed_metrics


def calculate_metric_delta(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> dict[str, float]:

    return {
        metric_name: (
            metrics_after[metric_name]
            - metrics_before[metric_name]
        )
        for metric_name in metrics_before
    }


def calculate_speculative_evidence(
    request_id: str,
    metric_delta: dict[str, float],
    total_request_seconds: float,
) -> dict:

    verification_steps = metric_delta[
        "verification_steps"
    ]

    proposed_draft_tokens = metric_delta[
        "draft_tokens"
    ]

    accepted_draft_tokens = metric_delta[
        "accepted_tokens"
    ]

    rejected_draft_tokens = max(
        proposed_draft_tokens
        - accepted_draft_tokens,
        0.0,
    )

    if proposed_draft_tokens > 0:
        draft_acceptance_rate = (
            accepted_draft_tokens
            / proposed_draft_tokens
        )
    else:
        draft_acceptance_rate = 0.0

    if verification_steps > 0:
        mean_tokens_per_verification = (
            1.0
            + accepted_draft_tokens
            / verification_steps
        )
    else:
        mean_tokens_per_verification = 1.0

    return {
        "request_id":
            request_id,

        "verification_steps":
            verification_steps,

        "proposed_draft_tokens":
            proposed_draft_tokens,

        "accepted_draft_tokens":
            accepted_draft_tokens,

        "rejected_draft_tokens":
            rejected_draft_tokens,

        "draft_acceptance_rate_percent":
            draft_acceptance_rate * 100.0,

        "mean_tokens_per_verification":
            mean_tokens_per_verification,

        "total_request_seconds":
            total_request_seconds,
    }


def run_one_case(
    request_spec: RequestSpec,
    vllm_url: str,
) -> dict:

    metrics_url = (
        f"{vllm_url}/metrics"
    )

    metrics_before = fetch_speculative_metrics(
        metrics_url=metrics_url,
    )

    experiment_start_time = time.perf_counter()

    request_result, _ = send_streaming_request(
        request_spec=request_spec,
        experiment_start_time=experiment_start_time,
        vllm_url=vllm_url,
    )

    metrics_after = fetch_speculative_metrics(
        metrics_url=metrics_url,
    )

    metric_delta = calculate_metric_delta(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )

    return calculate_speculative_evidence(
        request_id=request_spec.request_id,
        metric_delta=metric_delta,
        total_request_seconds=(
            request_result.total_request_seconds
        ),
    )


def write_evidence(
    evidence_rows: list[dict],
) -> None:

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIRECTORY
        / "speculative-decoding-evidence.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=evidence_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(evidence_rows)


def print_evidence(
    evidence_rows: list[dict],
) -> None:

    print()
    print("Speculative-decoding evidence")
    print("-----------------------------")

    print(
        f"{'Request':<16}"
        f"{'Proposed':<12}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Accept %':<12}"
        f"{'Tokens/verify':<14}"
    )

    print("-" * 78)

    for evidence in evidence_rows:

        print(
            f"{evidence['request_id']:<16}"
            f"{evidence['proposed_draft_tokens']:<12.0f}"
            f"{evidence['accepted_draft_tokens']:<12.0f}"
            f"{evidence['rejected_draft_tokens']:<12.0f}"
            f"{evidence['draft_acceptance_rate_percent']:<11.1f}%"
            f"{evidence['mean_tokens_per_verification']:<14.2f}"
        )

    print()


def get_requests_for_case(
    experiment_case: str,
) -> list[RequestSpec]:

    if experiment_case == "predictable":
        return [
            build_predictable_request()
        ]

    if experiment_case == "unpredictable":
        return [
            build_unpredictable_request()
        ]

    if experiment_case == "all":
        return [
            build_predictable_request(),
            build_unpredictable_request(),
        ]

    raise ValueError(
        f"Unsupported case: {experiment_case}"
    )


def print_prediction() -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        "PREDICTABLE already contains a repeated sequence. "
        "The n-gram proposer should frequently find useful "
        "continuations in the existing context."
    )

    print(
        "UNPREDICTABLE asks for novel prose. Proposed n-grams "
        "should match the target model less frequently."
    )

    print()
    print(
        "We do not assume the prediction is correct. "
        "Accepted/rejected-token counters decide."
    )
    print()


def print_server_requirement() -> None:

    print("Required server configuration")
    print("-----------------------------")

    print(
        "Start Qwen3 with n-gram speculative decoding:"
    )

    print()

    print(
        """--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":1,"prompt_lookup_max":4}'"""
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-07: Observe speculative decoding as "
            "proposal -> verification -> acceptance/rejection."
        )
    )

    argument_parser.add_argument(
        "--case",
        choices=[
            "predictable",
            "unpredictable",
            "all",
        ],
        default="all",
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

    request_specs = get_requests_for_case(
        experiment_case=arguments.case,
    )

    print()
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Case       : {arguments.case}")

    print_prediction()
    print_server_requirement()

    if arguments.dry_run:

        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.case,
        )

        return

    evidence_rows = []

    for request_spec in request_specs:

        print(
            f"Running {request_spec.request_id}..."
        )

        evidence = run_one_case(
            request_spec=request_spec,
            vllm_url=arguments.vllm_url,
        )

        evidence_rows.append(
            evidence
        )

    write_evidence(
        evidence_rows=evidence_rows,
    )

    print_evidence(
        evidence_rows=evidence_rows,
    )

    print(
        "Evidence saved under:"
    )

    print(
        RESULTS_DIRECTORY
        / "speculative-decoding-evidence.csv"
    )


if __name__ == "__main__":
    main()
