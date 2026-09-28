from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import evaluation
from comparison import compare_runs
from credentials import (
    credentials_path,
    load_credentials,
    model_credentials,
    save_model_credentials,
)
from model_registry import MODEL_REGISTRY, SOURCE, doctor, resolve_model

console = Console()


def _prepare_http_credentials(
    args: argparse.Namespace,
    *,
    force: bool = False,
) -> None:
    model, entry = resolve_model(args.model)
    if not entry.get("requires_endpoint"):
        if force:
            raise ValueError(f"{model} does not use an HTTP endpoint")
        return

    saved = model_credentials(model)
    supplied_endpoint = getattr(args, "endpoint", None)
    endpoint = supplied_endpoint or saved.get("endpoint")
    if force or not endpoint:
        if not console.is_terminal:
            raise RuntimeError(
                f"No endpoint saved for {model}. Run: srb configure {model}"
            )
        endpoint = Prompt.ask(
            "[bold cyan]System One endpoint URL[/]",
            default=endpoint,
            console=console,
        )
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Endpoint must be an http:// or https:// URL")

    endpoint_changed = endpoint != saved.get("endpoint")
    ask_for_key = force or "api_key" not in saved or endpoint_changed
    api_key = saved.get("api_key", "")
    if ask_for_key and console.is_terminal:
        entered = Prompt.ask(
            "[bold cyan]API key[/] [dim](empty keeps saved; - clears)[/]",
            default="",
            password=True,
            show_default=False,
            console=console,
        )
        if entered == "-":
            api_key = ""
        elif entered or "api_key" not in saved:
            api_key = entered

    if endpoint_changed or api_key != saved.get("api_key"):
        path = save_model_credentials(model, endpoint, api_key)
        console.print(f"[green]✓ Credentials saved[/] [dim]{path}[/]")
    args.endpoint = endpoint
    args.api_key = api_key


def _configure(args: argparse.Namespace) -> int:
    _prepare_http_credentials(args, force=True)
    console.print(
        Panel.fit(
            f"[green]Saved configuration for[/] [bold]{args.model}[/]\n"
            f"[dim]{credentials_path()} · mode 0600[/]",
            title="[bold green]✓ Configured[/]",
            border_style="green",
        )
    )
    return 0


def _models_list(_args: argparse.Namespace) -> int:
    saved = load_credentials()
    table = Table(
        title="Decision Index Models",
        title_style="bold cyan",
        header_style="bold white",
        border_style="bright_black",
        row_styles=("", "dim"),
        show_lines=False,
    )
    table.add_column("Model", style="bold")
    table.add_column("Adapter", style="cyan")
    table.add_column("Probabilities", justify="center")
    table.add_column("Endpoint", justify="center")
    for model, entry in sorted(MODEL_REGISTRY.items()):
        probability = entry["probabilities"]
        probability_style = "green" if probability == "native" else "yellow"
        if entry.get("requires_endpoint"):
            endpoint = (
                "[green]saved[/]"
                if saved.get(model, {}).get("endpoint")
                else "[yellow]required[/]"
            )
        else:
            endpoint = "[dim]local[/]"
        table.add_row(
            model,
            entry["adapter"],
            f"[{probability_style}]{probability}[/]",
            endpoint,
        )
    console.print(table)
    console.print(f"[dim]Source: {SOURCE}[/]")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    if not args.endpoint:
        args.endpoint = model_credentials(args.model).get("endpoint")
    checks = doctor(args.model, args.endpoint)
    table = Table(
        title=f"Runtime Check · {args.model}",
        title_style="bold cyan",
        border_style="bright_black",
        header_style="bold white",
    )
    table.add_column("Status", justify="center")
    table.add_column("Check")
    table.add_column("Details", style="dim")
    for name, ok, detail in checks:
        status = (
            "[bold green]✓ ready[/]"
            if ok
            else "[bold red]✗ missing[/]"
        )
        table.add_row(status, name, detail)
    console.print(table)
    return 0 if all(ok for _, ok, _ in checks) else 1


def _run(args: argparse.Namespace) -> int:
    _prepare_http_credentials(args)
    metrics = evaluation.evaluate(args)
    output = Path(args.output)
    details = Table.grid(padding=(0, 2))
    details.add_column(style="cyan")
    details.add_column(style="white")
    details.add_row("Metric groups", str(len(metrics)))
    details.add_row("Metrics", str(output / "metrics.json"))
    details.add_row("Predictions", str(output / "predictions.jsonl"))
    details.add_row("Run metadata", str(output / "run.json"))
    console.print(
        Panel(
            details,
            title="[bold green]✓ Run complete[/]",
            border_style="green",
        )
    )
    return 0

def _metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _compare(args: argparse.Namespace) -> int:
    rows = compare_runs(args.results)
    table = Table(
        title="Benchmark Comparison",
        title_style="bold cyan",
        header_style="bold white",
        border_style="bright_black",
        row_styles=("", "dim"),
    )
    for name in (
        "System",
        "Text",
        "Representation",
        "Track",
        "Difficulty",
        "K",
        "Accuracy",
        "NLL",
        "ECE",
        "P50 ms",
        "Forwards",
    ):
        numeric = {"K", "Accuracy", "NLL", "ECE", "P50 ms", "Forwards"}
        table.add_column(
            name,
            justify="right" if name in numeric else "left",
        )
    for row in rows:
        table.add_row(
            row["system"],
            row["text_mode"],
            row["representation"],
            row["community_track"],
            row["difficulty"],
            str(row["k"]),
            _metric(row["accuracy"]),
            _metric(row["nll"]),
            _metric(row["ece_15"]),
            _metric(row["latency_ms_p50"]),
            _metric(row["mean_forward_count"]),
        )
    console.print(table)
    if args.output:
        output = Path(args.output)
        output.write_text(json.dumps(rows, indent=2) + "\n")
        console.print(f"[green]✓ Comparison written[/] [dim]{output}[/]")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="srb",
        description="Social Routing Bench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    models = subparsers.add_parser(
        "models",
        help="Inspect supported Decision Index models",
    )
    model_commands = models.add_subparsers(
        dest="models_command",
        required=True,
    )
    compare = subparsers.add_parser(
        "compare",
        help="Compare compatible benchmark result directories",
    )
    compare.add_argument("results", nargs="+")
    compare.add_argument("--output")
    compare.set_defaults(handler=_compare)

    listing = model_commands.add_parser("list")
    listing.set_defaults(handler=_models_list)

    configure = subparsers.add_parser(
        "configure",
        help="Save an HTTP endpoint and API key for a model",
    )
    configure.add_argument("model")
    configure.add_argument("--endpoint")
    configure.set_defaults(handler=_configure)

    diagnosis = subparsers.add_parser("doctor", help="Check a model runtime")
    diagnosis.add_argument("model")
    diagnosis.add_argument("--endpoint")
    diagnosis.set_defaults(handler=_doctor)

    run = subparsers.add_parser(
        "run",
        help="Run the subreddit routing benchmark",
    )
    run.add_argument("model")
    run.add_argument("--dataset", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument(
        "--task-config",
        default="tasks/subreddit_dynamic/task.yaml",
    )
    run.add_argument("--posts-per-community", type=int)
    run.add_argument(
        "--text-modes",
        nargs="+",
        choices=("title", "title+selftext"),
    )
    run.add_argument("--adapter")
    run.add_argument("--endpoint")
    run.add_argument("--command-worker")
    run.set_defaults(handler=_run)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        code = args.handler(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        code = 130
    except (RuntimeError, ValueError) as error:
        console.print(
            Panel.fit(
                str(error),
                title="[bold red]Error[/]",
                border_style="red",
            )
        )
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
