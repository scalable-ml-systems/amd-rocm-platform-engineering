"""Execute one command under an experiment contract and capture evidence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from rocm_platform_engineering.contracts.experiment_contract import (
    load_experiment_contract,
)
from rocm_platform_engineering.contracts.experiment_result import (
    ExperimentResult,
    RunStatus,
)

DEFAULT_RESULTS_DIRECTORY = Path("benchmark-results") / "summaries"


def current_utc_time() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def get_git_revision(repository_root: Path) -> str:
    """Return the current full Git revision."""
    completed_process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed_process.stdout.strip().lower()


def create_run_id(
    started_at_utc: datetime,
    git_revision: str,
) -> str:
    """Create the standard run identifier."""
    timestamp = started_at_utc.strftime("%Y%m%dT%H%M%SZ")
    short_revision = git_revision[:8]
    return f"run-{timestamp}-{short_revision}"


def write_json_file(
    output_path: Path,
    data: dict,
) -> None:
    """Write formatted JSON to disk."""
    output_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def build_environment_record(
    command: Sequence[str],
    repository_root: Path,
) -> dict:
    """Create a small record of the command execution environment."""
    return {
        "command": list(command),
        "working_directory": str(repository_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def execute_experiment(
    experiment_path: Path,
    command: Sequence[str],
    repository_root: Path,
    results_directory: Path = DEFAULT_RESULTS_DIRECTORY,
) -> Path:
    """Execute one command and write its experiment evidence."""
    if not command:
        raise ValueError("An experiment command is required")

    experiment = load_experiment_contract(experiment_path)
    git_revision = get_git_revision(repository_root)
    started_at_utc = current_utc_time()

    run_id = create_run_id(
        started_at_utc=started_at_utc,
        git_revision=git_revision,
    )

    run_directory = (
        repository_root
        / results_directory
        / experiment.experiment_id
        / run_id
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    environment_path = run_directory / "environment.json"
    stdout_path = run_directory / "stdout.log"
    stderr_path = run_directory / "stderr.log"
    manifest_path = run_directory / "run-manifest.json"

    write_json_file(
        environment_path,
        build_environment_record(
            command=command,
            repository_root=repository_root,
        ),
    )

    exit_code = 1
    status = RunStatus.FAIL

    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed_process = subprocess.run(
                list(command),
                cwd=repository_root,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                text=True,
            )
            exit_code = completed_process.returncode
            status = (
                RunStatus.PASS
                if exit_code == 0
                else RunStatus.FAIL
            )
    except KeyboardInterrupt:
        exit_code = 130
        status = RunStatus.STOPPED

        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            stderr_file.write(
                "\nExperiment stopped by keyboard interrupt.\n"
            )
    finally:
        finished_at_utc = current_utc_time()

        result = ExperimentResult(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            hardware_profile_id=experiment.hardware_profile_id,
            software_stack_id=experiment.software_stack_id,
            workload_profile_id=experiment.workload_profile_id,
            git_revision=git_revision,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            status=status,
            exit_code=exit_code,
            summary_metrics=[],
            evidence_paths=[
                "environment.json",
                "stdout.log",
                "stderr.log",
            ],
            notes=[],
        )

        manifest_path.write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    return manifest_path


def parse_command(command_parts: Sequence[str]) -> list[str]:
    """Remove the optional argument separator from a command."""
    command = list(command_parts)

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        raise ValueError(
            "Provide the experiment command after '--'"
        )

    return command


def main() -> None:
    """Run one command under an experiment contract."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute one command and capture a minimal "
            "experiment evidence bundle."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to the experiment YAML contract.",
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )

    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help="Results directory relative to the repository root.",
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute after '--'.",
    )

    arguments = parser.parse_args()

    repository_root = arguments.repository_root.resolve()

    experiment_path = arguments.experiment
    if not experiment_path.is_absolute():
        experiment_path = repository_root / experiment_path

    try:
        command = parse_command(arguments.command)
        manifest_path = execute_experiment(
            experiment_path=experiment_path,
            command=command,
            repository_root=repository_root,
            results_directory=arguments.results_directory,
        )
    except (
        FileNotFoundError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        parser.error(str(error))

    print(f"Experiment manifest: {manifest_path}")


if __name__ == "__main__":
    main()
