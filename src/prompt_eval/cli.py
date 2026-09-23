"""prompt-eval command line."""

from __future__ import annotations

from pathlib import Path

import typer

from prompt_eval.dotenv import load_dotenv
from prompt_eval.errors import PromptEvalError
from prompt_eval.llm import build_client
from prompt_eval.report import dumps_payload, render_text, to_payload
from prompt_eval.runner import execute
from prompt_eval.store import DEFAULT_RESULTS_PATH, save_results
from prompt_eval.suite import load_suite, select_cases

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def callback() -> None:
    """Run a prompt evaluation suite against a local model."""


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", help="YAML or JSON suite file."),
    case: str | None = typer.Option(None, "--case", help="Run one case id from the suite."),
    json_report: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List cases and do not call a model."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the LLM judge. Score with deterministic checks only."),
    results: Path | None = typer.Option(None, "--results", help="Write the results JSON to this path."),
    no_save: bool = typer.Option(False, "--no-save", help="Do not write a results file."),
    provider: str | None = typer.Option(None, "--provider", help="ollama or openai. Overrides PROMPT_EVAL_PROVIDER."),
    model: str | None = typer.Option(None, "--model", help="Model name for this command."),
) -> None:
    """Run every case in a suite. Exits 1 when a scored case fails."""
    try:
        _run(
            suite=suite,
            case=case,
            json_report=json_report,
            dry_run=dry_run,
            no_judge=no_judge,
            results=results,
            no_save=no_save,
            provider=provider,
            model=model,
        )
    except PromptEvalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _run(
    *,
    suite: Path,
    case: str | None,
    json_report: bool,
    dry_run: bool,
    no_judge: bool,
    results: Path | None,
    no_save: bool,
    provider: str | None,
    model: str | None,
) -> None:
    if results is not None and no_save:
        raise PromptEvalError("Pass either --results or --no-save, not both.")
    load_dotenv()
    loaded = load_suite(suite)
    selected = select_cases(loaded, case)
    client = None
    try:
        if not dry_run:
            client = build_client(provider, model)
        report = execute(
            suite=loaded,
            suite_path=suite,
            cases=selected,
            client=client,
            no_judge=no_judge,
            dry_run=dry_run,
        )
    finally:
        if client is not None:
            client.close()

    results_path = _results_path(dry_run=dry_run, results=results, no_save=no_save)
    payload = to_payload(report, None if results_path is None else str(results_path))
    if results_path is not None:
        save_results(results_path, payload)
        typer.echo(f"Saved {results_path}", err=True)
    elif dry_run and results is not None:
        typer.echo("warning: dry run does not write results.", err=True)

    if json_report:
        typer.echo(dumps_payload(payload), nl=False)
    else:
        typer.echo(render_text(report), nl=False)

    if report.aborted:
        typer.echo(f"error: {report.aborted}", err=True)
        raise typer.Exit(code=2)
    if not dry_run and not report.ok:
        raise typer.Exit(code=1)


def _results_path(*, dry_run: bool, results: Path | None, no_save: bool) -> Path | None:
    if dry_run or no_save:
        return None
    if results is not None:
        return results
    return DEFAULT_RESULTS_PATH
