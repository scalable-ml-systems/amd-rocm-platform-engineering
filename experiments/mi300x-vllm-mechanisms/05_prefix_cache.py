from __future__ import annotations

import argparse
import csv
import time
import urllib.request

from pathlib import Path

from common import (
    RequestSpec,
    make_approximate_token_prompt,
    print_dry_run,
    send_streaming_request,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "05-prefix-cache"

RESULTS_DIRECTORY = Path(
    "results/05-prefix-cache"
)


PREFIX_METRIC_NAMES = {
    "prefix_queries": {
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
    },
    "prefix_hits": {
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    },
}


def build_shared_prefix() -> str:
    """
    Create one deterministic long prefix.

    Both R1 and R2 use this exact text.
    """

    repeated_context = make_approximate_token_prompt(
        approximate_token_count=2048,
        repeated_text=(
            "GPU inference memory scheduler attention "
            "model request token cache execution "
        ),
    )

    return (
        "SYSTEM NOTES\n\n"
        "GPU infrastructure observations follow.\n\n"
        + repeated_context
        + "\n\nEND SYSTEM NOTES\n\n"
    )


def build_mutated_prefix(
    shared_prefix: str,
) -> str:
    """
    Change one early word.

    Because prefix-cache block hashes depend on the preceding
    prefix, an early change should prevent downstream blocks
    from matching the original cached prefix.
    """

    return shared_prefix.replace(
        "GPU infrastructure",
        "CPU infrastructure",
        1,
    )


def build_requests() -> list[RequestSpec]:

    shared_prefix = build_shared_prefix()

    mutated_prefix = build_mutated_prefix(
        shared_prefix=shared_prefix,
    )

    return [
        RequestSpec(
            request_id="R1-COLD",
            prompt=(
                shared_prefix
                + "Question: summarize the role of memory bandwidth."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),

        RequestSpec(
            request_id="R2-REUSE",
            prompt=(
                shared_prefix
                + "Question: summarize the role of request scheduling."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),

        RequestSpec(
            request_id="R3-MUTATED",
            prompt=(
                mutated_prefix
                + "Question: summarize the role of attention."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),
    ]


def fetch_prefix_cache_metrics(
    metrics_url: str,
) -> dict[str, float]:
    """
    Read only prefix-cache counters from vLLM /metrics.

    The accepted metric-name variants make this tolerant of
    minor naming differences across vLLM releases.
    """

    metric_values = {
        "prefix_queries": 0.0,
        "prefix_hits": 0.0,
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

        fields = line.split()

        if len(fields) < 2:
            continue

        metric_name = fields[0].split("{", 1)[0]

        try:
            metric_value = float(fields[1])
        except ValueError:
            continue

        for logical_name, accepted_names in (
            PREFIX_METRIC_NAMES.items()
        ):
            if metric_name in accepted_names:
                metric_values[logical_name] += metric_value

    return metric_values


def calculate_metric_delta(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> dict[str, float]:

    return {
        metric_name: (
            metrics_after.get(metric_name, 0.0)
            - metrics_before.get(metric_name, 0.0)
        )
        for metric_name in metrics_after
    }


def run_one_request(
    request_spec: RequestSpec,
    experiment_start_time: float,
    vllm_url: str,
) -> tuple:

    request_result, event_records = (
        send_streaming_request(
            request_spec=request_spec,
            experiment_start_time=experiment_start_time,
            vllm_url=vllm_url,
        )
    )

    for event_record in event_records:
        event_record.experiment_name = EXPERIMENT_NAME
        event_record.experiment_mode = "prefix-reuse"

    return request_result, event_records


def run_prefix_cache_experiment(
    vllm_url: str,
) -> None:

    request_specs = build_requests()

    metrics_url = f"{vllm_url}/metrics"

    experiment_start_time = time.perf_counter()

    all_request_results = []
    all_event_records = []
    cache_evidence = []

    for request_spec in request_specs:

        metrics_before = fetch_prefix_cache_metrics(
            metrics_url=metrics_url,
        )

        request_result, event_records = (
            run_one_request(
                request_spec=request_spec,
                experiment_start_time=experiment_start_time,
                vllm_url=vllm_url,
            )
        )

        metrics_after = fetch_prefix_cache_metrics(
            metrics_url=metrics_url,
        )

        metric_delta = calculate_metric_delta(
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

        queried_tokens = metric_delta[
            "prefix_queries"
        ]

        hit_tokens = metric_delta[
            "prefix_hits"
        ]

        if queried_tokens > 0:
            hit_rate_percent = (
                hit_tokens
                / queried_tokens
                * 100.0
            )
        else:
            hit_rate_percent = 0.0

        cache_evidence.append(
            {
                "request_id":
                    request_spec.request_id,

                "prefix_query_tokens":
                    queried_tokens,

                "prefix_hit_tokens":
                    hit_tokens,

                "prefix_hit_rate_percent":
                    hit_rate_percent,

                "time_to_first_token_seconds":
                    request_result.time_to_first_token_seconds,

                "total_request_seconds":
                    request_result.total_request_seconds,
            }
        )

        all_request_results.append(
            request_result
        )

        all_event_records.extend(
            event_records
        )

    save_results(
        request_results=all_request_results,
        event_records=all_event_records,
        cache_evidence=cache_evidence,
    )

    print_summary(
        cache_evidence=cache_evidence,
    )


def save_results(
    request_results,
    event_records,
    cache_evidence: list[dict],
) -> None:

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        records=request_results,
        output_path=(
            RESULTS_DIRECTORY
            / "request-results.csv"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            RESULTS_DIRECTORY
            / "client-events.jsonl"
        ),
    )

    with (
        RESULTS_DIRECTORY
        / "prefix-cache-evidence.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=cache_evidence[0].keys(),
        )

        writer.writeheader()
        writer.writerows(cache_evidence)


def print_summary(
    cache_evidence: list[dict],
) -> None:

    print()
    print("Prefix-cache evidence")
    print("---------------------")

    print(
        f"{'Request':<16}"
        f"{'Queried':<12}"
        f"{'Hits':<12}"
        f"{'Hit rate':<12}"
        f"{'TTFT(s)':<12}"
    )

    print("-" * 64)

    for evidence in cache_evidence:

        time_to_first_token = (
            evidence[
                "time_to_first_token_seconds"
            ]
        )

        if time_to_first_token is None:
            formatted_ttft = "N/A"
        else:
            formatted_ttft = (
                f"{time_to_first_token:.4f}"
            )

        print(
            f"{evidence['request_id']:<16}"
            f"{evidence['prefix_query_tokens']:<12.0f}"
            f"{evidence['prefix_hit_tokens']:<12.0f}"
            f"{evidence['prefix_hit_rate_percent']:<11.1f}%"
            f"{formatted_ttft:<12}"
        )

    print()


def print_prediction() -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        "R1-COLD has no previously cached prefix, so it should "
        "have little or no prefix-cache reuse."
    )

    print(
        "R2-REUSE uses the identical long prefix, so most full "
        "prefix blocks should be cache hits."
    )

    print(
        "R3-MUTATED changes one word near the beginning. "
        "Because later block hashes depend on their preceding prefix, "
        "reuse should fall sharply after that change."
    )

    print()


def print_server_requirement() -> None:

    print("Required server configuration")
    print("-----------------------------")

    print(
        "Qwen3 server with automatic prefix caching enabled:"
    )

    print()
    print(
        "  --enable-prefix-caching"
    )
    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-05: Observe automatic prefix-cache reuse "
            "and deliberately break the shared prefix."
        )
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
    print(f"Experiment : {EXPERIMENT_NAME}")

    print_prediction()
    print_server_requirement()

    if arguments.dry_run:

        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode="prefix-reuse",
        )

        return

    run_prefix_cache_experiment(
        vllm_url=arguments.vllm_url,
    )


if __name__ == "__main__":
    main()
