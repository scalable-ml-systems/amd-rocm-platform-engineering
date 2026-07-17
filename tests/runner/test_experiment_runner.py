"""Focused tests for the minimal experiment runner."""

from pathlib import Path

from rocm_platform_engineering.contracts.experiment_result import (
    load_experiment_result,
)
from rocm_platform_engineering.runner.experiment_runner import (
    execute_experiment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "examples"
    / "exp-p02-003-pinned-memory-transfer"
    / "experiment.yaml"
)


def test_successful_command_writes_passing_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = execute_experiment(
        experiment_path=EXPERIMENT_PATH,
        command=[
            "python",
            "-c",
            "print('runner success')",
        ],
        repository_root=REPOSITORY_ROOT,
        results_directory=tmp_path,
    )

    result = load_experiment_result(manifest_path)

    assert result.status == "pass"
    assert result.exit_code == 0
    assert (
        manifest_path.parent / "stdout.log"
    ).read_text(encoding="utf-8").strip() == "runner success"


def test_failed_command_preserves_failure_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = execute_experiment(
        experiment_path=EXPERIMENT_PATH,
        command=[
            "python",
            "-c",
            "import sys; sys.exit(2)",
        ],
        repository_root=REPOSITORY_ROOT,
        results_directory=tmp_path,
    )

    result = load_experiment_result(manifest_path)

    assert result.status == "fail"
    assert result.exit_code == 2
    assert (manifest_path.parent / "stderr.log").is_file()
    assert (manifest_path.parent / "environment.json").is_file()
