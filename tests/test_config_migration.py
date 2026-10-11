"""Old project.yaml files carry old defaults; opening one migrates the values that changed."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.config import CONFIG_VERSION, ProjectConfig, migrate
from gossan.core.project import Project

runner = CliRunner()


def old_project(tmp_path, *, cv_buffer_m, with_version=False) -> Path:
    """A project folder whose project.yaml was written by an earlier Gossan."""
    root = tmp_path / "old"
    root.mkdir()
    config = ProjectConfig(name="old", crs="EPSG:32611")
    data = config.model_dump(mode="json")
    del data["config_version"]  # files before versioning have no such key
    data["target"]["cv_buffer_m"] = cv_buffer_m
    del data["geochem"]  # sections added later are simply absent
    if with_version:
        data["config_version"] = 1
    (root / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def test_migrate_function_changes_only_the_old_default():
    data, notes, start = migrate({"target": {"cv_buffer_m": 0.0}})
    assert start == 1 and data["config_version"] == CONFIG_VERSION and data["target"]["cv_buffer_m"] is None
    assert len(notes) == 1 and "flatters" in notes[0] and "project.yaml" in notes[0]
    data, notes, _ = migrate({"target": {"cv_buffer_m": 1500.0}})
    assert notes == [] and data["target"]["cv_buffer_m"] == 1500.0  # a deliberate value is kept
    data, notes, start = migrate({"config_version": CONFIG_VERSION, "target": {"cv_buffer_m": 0.0}})
    assert notes == [] and start == CONFIG_VERSION and data["target"]["cv_buffer_m"] == 0.0  # 0 chosen on purpose stays


def test_opening_an_old_project_updates_the_file_and_logs_it(tmp_path):
    root = old_project(tmp_path, cv_buffer_m=0.0)
    project = Project(root)
    assert project.config.target.cv_buffer_m is None and project.migration_notes
    saved = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))
    assert saved["config_version"] == CONFIG_VERSION and saved["target"]["cv_buffer_m"] is None
    (record,) = [r for r in project.provenance.read() if r.operation == "migrate_config"]
    assert record.parameters["from_version"] == 1 and record.parameters["to_version"] == CONFIG_VERSION
    assert "cv_buffer_m" in record.parameters["notes"][0]
    again = Project(root)  # the second open has nothing left to migrate
    assert again.migration_notes == [] and len([r for r in again.provenance.read() if r.operation == "migrate_config"]) == 1


def test_a_deliberate_value_survives_and_only_the_version_is_stamped(tmp_path):
    root = old_project(tmp_path, cv_buffer_m=1500.0, with_version=True)
    project = Project(root)
    assert project.config.target.cv_buffer_m == 1500.0 and project.migration_notes == []
    assert yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))["config_version"] == CONFIG_VERSION
    assert not [r for r in project.provenance.read() if r.operation == "migrate_config"]  # nothing changed: no log


def test_zero_set_after_the_migration_is_respected(tmp_path):
    root = old_project(tmp_path, cv_buffer_m=0.0)
    project = Project(root)
    project.config.target.cv_buffer_m = 0.0  # the user pins the old behaviour on purpose
    project.config.save(root)
    again = Project(root)
    assert again.config.target.cv_buffer_m == 0.0 and again.migration_notes == []


def test_new_projects_are_current_and_the_cli_announces_a_migration(aoi_file):
    assert runner.invoke(app, ["new", "fresh", "--aoi", str(aoi_file)]).exit_code == 0
    assert yaml.safe_load(Path("fresh/project.yaml").read_text(encoding="utf-8"))["config_version"] == CONFIG_VERSION
    assert Project(Path("fresh")).migration_notes == []

    root = old_project(Path.cwd(), cv_buffer_m=0.0)
    (root / "aoi.gpkg").write_bytes((Path("fresh") / "aoi.gpkg").read_bytes())
    result = runner.invoke(app, ["analyze", str(root), "--steps", "inventory"])
    assert "project.yaml updated to settings version" in result.output and "flatters the scores" in result.output


def test_version_3_makes_metallic_labels_the_default_for_projects_still_on_the_old_default():
    data, notes, start = migrate({"config_version": 2, "target": {"occurrence_class": "all", "cv_buffer_m": None}})
    assert start == 2 and data["target"]["occurrence_class"] == "auto" and len(notes) == 1
    assert "validated BC areas" in notes[0] and "'all'" in notes[0]  # says why, and how to go back
    for chosen in ("metallic", "industrial", "specialty"):
        data, notes, _ = migrate({"config_version": 2, "target": {"occurrence_class": chosen}})
        assert data["target"]["occurrence_class"] == chosen and notes == []  # a deliberate class is kept
    data, notes, _ = migrate({"config_version": 3, "target": {"occurrence_class": "all"}})
    assert data["target"]["occurrence_class"] == "all" and notes == []  # "all" chosen after the change stays


def test_an_old_project_opens_on_auto_and_logs_the_change(tmp_path):
    root = tmp_path / "v2"
    root.mkdir()
    data = ProjectConfig(name="v2", crs="EPSG:32611").model_dump(mode="json")
    data["config_version"] = 2
    data["target"]["occurrence_class"] = "all"
    (root / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    project = Project(root)
    assert project.config.target.occurrence_class == "auto" and any("occurrence_class" in n for n in project.migration_notes)
    (record,) = [r for r in project.provenance.read() if r.operation == "migrate_config"]
    assert record.parameters["from_version"] == 2 and record.parameters["to_version"] == CONFIG_VERSION == 3
