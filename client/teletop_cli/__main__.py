"""teletop CLI entry point."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """teletop — remote ESP32 flash & monitor."""


@cli.command()
def status() -> None:
    """Show server connection status (placeholder)."""
    console.print("[yellow]not implemented[/yellow]")


if __name__ == "__main__":
    cli()
