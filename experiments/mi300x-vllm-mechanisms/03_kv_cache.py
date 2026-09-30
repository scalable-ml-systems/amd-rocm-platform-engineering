from __future__ import annotations

import argparse
import csv
import threading
import time
import urllib.request

from pathlib import Path

from common import (
    RequestSpec,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "03-kv-cache"

RESULTS_DIRECTORY = Path(
    "results/03-kv-cache"
)

METRIC_SAMPLE_INTERVAL_SECONDS = 0.05

METRICS_TO_OBSERVE = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions",
}


def build_requests() -> list[RequestSpec]:
    """
    Four requests generate long outputs concurrently.

    As output tokens accumulate, each request needs more KV-cache
    storage for its growing sequence.
    """

    return [
        RequestSpec(
            request_id="R1",
            prompt=(
                "Explain GPU memory hierarchy in detail. "
                "Continue systematically through each component."
            ),
            max_output_tokens=768,
            start_delay_seconds=0.00,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R2",
            prompt=(
                "Explain how an inference scheduler works in detail. "
                "Continue systematically through each component."
            ),
            max_output_tokens=768,
            start_delay_seconds=0.05,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R3",
            prompt=(
                "Explain KV-cache management in detail. "
                "Continue systematically through each component."
            ),
            max_output_tokens=768,
            start_delay_seconds=0.10,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R4",
            prompt=(
                "Explain autoregressive decoding in detail. "
                "Continue systematically through each component."
            ),
            max_output_tokens=768,
            start_delay_seconds=0.15,
            temperature=0.0,
            ignore_eos=True,
        ),
    ]


def fetch_vllm_metrics(
    metrics_url: str,
) -> dict[str, float]:
    """
    Read the vLLM Prometheus metrics endpoint and return only
    the few metrics needed for this experiment.
    """

    observed_metrics: dict[str, float] = {}

    with urllib.request.urlopen(
        metrics_url,
        timeout=5,
    ) as response:

        metrics_text = response.read().decode("utf-8")

    for line in metrics_text.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        metric_and_value = line.split()

        if len(metric_and_value) < 2:
            continue

        metric_with_labels = metric_and_value[0]
        metric_value_text = metric_and_value[1]

        metric_name = metric_with_labels.split("{", 1)[0]

        if metric_name not in METRICS_TO_OBSERVE:
            continue

        try:
            metric_value = float(metric_value_text)
        except ValueError:
            continue

        # Some Prometheus metrics may appear more than once
        # with different labels. Summing is appropriate for
        # request/preemption counts. KV usage normally has one value.
        if metric_name in observed_metrics:
            observed_metrics[metric_name] += metric_value
        else:
            observed_metrics[metric_name] = metric_value

    return observed_metrics


def sample_metrics_until_stopped(
    metrics_url: str,
    experiment_start_time: float,
    stop_sampling: threading.Event,
    metric_samples: list[dict],
) -> None:
    """
    Poll vLLM while requests are running.

    This lets us see KV-cache occupancy grow over time rather
    than only observing its final value.
    """

    while not stop_sampling.is_set():

        elapsed_seconds = (
            time.perf_counter() - experiment_start_time
        )

        try:
            metrics = fetch_vllm_metrics(
                metrics_url=metrics_url,
            )

            metric_samples.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "kv_cache_usage_percent": (
                        metrics.get(
                            "vllm:kv_cache_usage_perc",
                            0.0,
                        )
                        * 100.0
                    ),
                    "running_requests": metrics.get(
                        "vllm:num_requests_running",
                        0.0,
                    ),
                    "waiting_requests": metrics.get(
                        "vllm:num_requests_waiting",
                        0.0,
                    ),
                    "preemptions": metrics.get(
                        "vllm:num_preemptions",
                        0.0,
                    ),
                }
            )

        except Exception as error:
            metric_samples.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "kv_cache_usage_percent": "",
                    "running_requests": "",
                    "waiting_requests": "",
                    "preemptions": "",
                    "error": str(error),
                }
            )

        stop_sampling.wait(
            METRIC_SAMPLE_INTERVAL_SECONDS
        )


def write_metric_samples(
    metric_samples: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metric_samples:
        return

    field_names = sorted(
        {
            field_name
            for sample in metric_samples
            for field_name in sample.keys()
        }
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(metric_samples)


def run_kv_cache_experiment(
    experiment_mode: str,
    vllm_url: str,
) -> None:

    request_specs = build_requests()

    metrics_url = (
        f"{vllm_url}/metrics"
    )

    experiment_start_time = time.perf_counter()

    stop_sampling = threading.Event()

    metric_samples: list[dict] = []

    metrics_thread = threading.Thread(
        target=sample_metrics_until_stopped,
        kwargs={
            "metrics_url": metrics_url,
            "experiment_start_time": experiment_start_time,
            "stop_sampling": stop_sampling,
            "metric_samples": metric_samples,
        },
        daemon=True,
    )

    metrics_thread.start()

    try:
        request_results, event_records = (
            run_staggered_requests(
                request_specs=request_specs,
                experiment_name=EXPERIMENT_NAME,
                experiment_mode=experiment_mode,
                vllm_url=vllm_url,
            )
        )

    finally:
        stop_sampling.set()
        metrics_thread.join()

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

    write_jsonl(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.jsonl"
        ),
    )

    write_metric_samples(
        metric_samples=metric_samples,
        output_path=(
            mode_results_directory
            / "kv-cache-timeline.csv"
        ),
    )

    print_summary(
        experiment_mode=experiment_mode,
        metric_samples=metric_samples,
    )


def print_summary(
    experiment_mode: str,
    metric_samples: list[dict],
) -> None:

    valid_samples = [
        sample
        for sample in metric_samples
        if isinstance(
            sample.get("kv_cache_usage_percent"),
            float,
        )
    ]

    print()
    print("KV-cache experiment summary")
    print("---------------------------")
    print(f"Mode: {experiment_mode}")

    if not valid_samples:
        print("No valid KV-cache metric samples collected.")
        return

    maximum_kv_usage = max(
        sample["kv_cache_usage_percent"]
        for sample in valid_samples
    )

    maximum_running_requests = max(
        sample["running_requests"]
        for sample in valid_samples
    )

    maximum_waiting_requests = max(
        sample["waiting_requests"]
        for sample in valid_samples
    )

    maximum_preemptions = max(
        sample["preemptions"]
        for sample in valid_samples
    )

    print(
        f"Maximum KV-cache usage : "
        f"{maximum_kv_usage:.1f}%"
    )

    print(
        f"Maximum running requests: "
        f"{maximum_running_requests:.0f}"
    )

    print(
        f"Maximum waiting requests: "
        f"{maximum_waiting_requests:.0f}"
    )

    print(
        f"Observed preemptions     : "
        f"{maximum_preemptions:.0f}"
    )

    print()


def print_prediction(
    experiment_mode: str,
) -> None:

    print()
    print("Prediction")
    print("----------")

    if experiment_mode == "baseline":

        print(
            "With the normal MI300X KV-cache allocation, "
            "cache usage should rise as the four sequences grow, "
            "but the requests should remain comfortably within capacity."
        )

    elif experiment_mode == "pressure":

        print(
            "With the number of GPU KV blocks deliberately restricted, "
            "cache usage should approach 100%. vLLM may queue or preempt "
            "requests because all active sequences cannot retain their "
            "required KV blocks simultaneously."
        )

    print()


def print_server_requirement(
    experiment_mode: str,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    if experiment_mode == "baseline":

        print(
            "Normal Qwen3 server. Let vLLM size the KV cache normally."
        )

    elif experiment_mode == "pressure":

        print(
            "Use the same Qwen3 server, but deliberately constrain "
            "the number of KV blocks:"
        )
        print()
        print("  --block-size 16")
        print("  --num-gpu-blocks-override 128")

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-03: Observe KV-cache growth and deliberately "
            "create KV-cache pressure."
        )
    )

    argument_parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "pressure",
        ],
        required=True,
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
    print(f"Mode       : {arguments.mode}")

    print_prediction(
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

    run_kv_cache_experiment(
        experiment_mode=arguments.mode,
        vllm_url=arguments.vllm_url,
    )


if __name__ == "__main__":
    main()
