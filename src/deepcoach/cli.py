"""deepcoach CLI: `deepcoach run <stage> --config <clip.yaml>`.

Each stage is independently runnable. The CLI is a thin dispatcher; it owns no
pipeline logic — it loads + validates the config and hands off to the stage's
run(config). This keeps stages independent and individually invocable.
"""

from __future__ import annotations

import typer

from .io.config import load_config
from .quality import report as quality_report
from .render import heatmap, overlay, pitchcontrol, radar
from .stages import (
    s0_ingest,
    s1_detect,
    s2_teams,
    s3_track,
    s4_homography,
    s5_project,
    s6_metrics,
)

app = typer.Typer(add_completion=False, help="deepcoach — soccer clip -> tactical-spatial data")

_STAGES = {
    "s0_ingest": s0_ingest,
    "s1_detect": s1_detect,
    "s2_teams": s2_teams,
    "s3_track": s3_track,
    "s4_homography": s4_homography,
    "s5_project": s5_project,
    "s6_metrics": s6_metrics,
    # convenience aliases
    "ingest": s0_ingest,
    "detect": s1_detect,
    "teams": s2_teams,
    "track": s3_track,
    "homography": s4_homography,
    "project": s5_project,
    "metrics": s6_metrics,
}


@app.command()
def run(
    stage: str = typer.Argument(..., help="stage name, e.g. s1_detect / detect"),
    config: str = typer.Option(..., "--config", "-c", help="path to per-clip YAML config"),
) -> None:
    """Run a single pipeline stage against a clip config."""
    if stage not in _STAGES:
        raise typer.BadParameter(f"unknown stage {stage!r}; choices: {sorted(_STAGES)}")
    cfg = load_config(config)
    typer.echo(f"[deepcoach] stage={stage} clip={cfg.clip_name()} config_hash={cfg.config_hash()}")
    _STAGES[stage].run(cfg)


@app.command()
def render(
    config: str = typer.Option(..., "--config", "-c", help="path to per-clip YAML config"),
    what: str = typer.Option("all", "--what", help="all | radar | overlay | heatmap | pitch_control"),
) -> None:
    """Render outputs enabled in the config (radar / overlay / heatmap / pitch_control)."""
    cfg = load_config(config)
    do = {w: (what in ("all", w)) for w in ("radar", "overlay", "heatmap", "pitch_control")}
    if do["overlay"] and cfg.render.overlay:
        overlay.render(cfg)
    if do["radar"] and cfg.render.radar:
        radar.render(cfg)
    if do["heatmap"] and cfg.render.heatmap:
        heatmap.render(cfg)
    if do["pitch_control"] and cfg.render.pitch_control:
        pitchcontrol.render(cfg)


@app.command()
def report(
    config: str = typer.Option(..., "--config", "-c", help="path to per-clip YAML config"),
) -> None:
    """Aggregate per-stage quality signals into one report (report.txt + report.json)."""
    quality_report.build_report(load_config(config))


@app.command()
def stages() -> None:
    """List available stages."""
    for name in ["s0_ingest", "s1_detect", "s2_teams", "s3_track", "s4_homography", "s5_project", "s6_metrics"]:
        typer.echo(name)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
