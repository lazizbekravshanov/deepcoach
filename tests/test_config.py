"""Config layer: the template validates, and config_hash is stable & order-independent."""

from __future__ import annotations

from deepcoach.io.config import ClipConfig, load_config


def test_template_loads_and_validates():
    cfg = load_config("configs/clip_template.yaml")
    assert cfg.pitch.length_m == 105
    assert cfg.teams.n_teams == 2
    assert cfg.homography.mode == "manual"
    assert cfg.clip_name() == "example"
    assert "centroid" in cfg.metrics.enabled


def test_config_hash_is_stable_and_16_hex():
    cfg = load_config("configs/clip_template.yaml")
    h1 = cfg.config_hash()
    h2 = ClipConfig.model_validate(cfg.model_dump()).config_hash()
    assert h1 == h2
    assert len(h1) == 16
    int(h1, 16)  # valid hex


def test_config_hash_changes_with_content():
    cfg = load_config("configs/clip_template.yaml")
    before = cfg.config_hash()
    cfg.pitch.length_m = 100.0
    assert cfg.config_hash() != before
