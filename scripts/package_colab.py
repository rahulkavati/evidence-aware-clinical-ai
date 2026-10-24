"""Build and validate a private Google Colab execution archive."""

from __future__ import annotations

import ast
import hashlib
import io
import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY_ROOT / "colab_package"
OUTPUT_ZIP = OUTPUT_DIR / "evidence_aware_colab.zip"
ENTRYPOINT = Path("experiments/run_batch_acquisition.py")
VALIDATION_MANIFEST = Path("outputs/splits/val.csv")
REQUIRED_MODEL_PATHS = (
    Path("outputs/models/xgboost.joblib"),
    Path("outputs/models/xgboost_sigmoid_calibrator.joblib"),
)
PATIENT_FILE_PREFIX = Path("src/data/raw/physionet2019")
EXPECTED_PATIENTS = 6050
REQUIREMENTS = """\
# Versions inspected in the repository environment. These pins preserve the
# dependency versions used to serialize and validate the saved model objects.
# Compatible Python range: 3.12-3.14; Google Colab's Python 3.12 runtime is supported.
numpy==2.5.4
pandas==3.0.6
scikit-learn==1.9.1
xgboost==3.4.1
joblib==1.6.0
matplotlib==3.11.2
"""
README = """\
# Evidence-Aware Clinical AI — Colab package

This private archive contains only the validation acquisition experiment, its
required Python modules, the saved XGBoost model and sigmoid calibrator, the
validation split manifest, and its 6,050 referenced PhysioNet patient records.
It does not contain training/test split manifests or test patient records.
Do not publish or share the archive: its validation PSV files are patient-level
clinical research data.

## Google Colab

Upload `evidence_aware_colab.zip` to the Colab session using the Files panel.
Then run:

```bash
!unzip -q /content/evidence_aware_colab.zip -d /content/evidence-aware-clinical-ai
%cd /content/evidence-aware-clinical-ai
!python --version
!pip install -r requirements-colab.txt
```

Run a deterministic 100-patient development benchmark:

```bash
!python -m experiments.run_batch_acquisition --development --sample-size 100
```

Run the full validation evaluation:

```bash
!python -m experiments.run_batch_acquisition
```

Aggregate reports and figures are written to
`experiments/results/acquisition/`. This experiment uses saved model artifacts;
it does not retrain XGBoost or refit the calibrator.

## Compatibility and interpretation

The package was assembled with Python {python_version}; it was validated in that
environment. The pinned versions inspected here are compatible with Python
3.12–3.14, including the Python 3.12 runtime commonly provided by Colab. Other
Python versions were not verified. Keeping these versions is recommended
because saved scikit-learn and XGBoost objects can be sensitive to
dependency-version changes.

This is retrospective historical information reveal, not clinical test
ordering. Eligibility uses an oracle assumption based on previously recorded
measurements; artificial masking can create out-of-distribution feature
patterns. These are validation/development results, not held-out test results,
and metric/risk changes alone do not establish clinical utility.
"""


class PackageError(RuntimeError):
    """Raised when source inputs or generated archive fail validation."""


def _relative_archive_name(value: str | Path) -> str:
    name = PurePosixPath(str(value).replace(os.sep, "/"))
    if (
        name.is_absolute()
        or not name.parts
        or any(part in ("", ".", "..") for part in name.parts)
        or "\\" in str(value)
        or "\x00" in str(value)
    ):
        raise PackageError("Unsafe path found while building the archive")
    return name.as_posix()


def _check_source_path(relative_path: Path) -> Path:
    archive_name = _relative_archive_name(relative_path.as_posix())
    del archive_name
    current = REPOSITORY_ROOT
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise PackageError("Symlink inputs are not permitted in the package")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise PackageError("Package input resolves outside the repository") from error
    if not resolved.is_file():
        raise PackageError("A required package input is not a regular file")
    return resolved


def _local_import_modules(module_path: Path) -> set[Path]:
    source = _check_source_path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=module_path.as_posix())
    local_modules: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported = [node.module]
        else:
            continue
        for module_name in imported:
            if not (module_name == "src" or module_name.startswith("src.")):
                continue
            module_path_base = Path(*module_name.split("."))
            file_candidate = module_path_base.with_suffix(".py")
            package_candidate = module_path_base / "__init__.py"
            if (REPOSITORY_ROOT / file_candidate).is_file():
                local_modules.add(file_candidate)
            elif (REPOSITORY_ROOT / package_candidate).is_file():
                local_modules.add(package_candidate)
            else:
                raise PackageError("A referenced local Python module could not be resolved")
    return local_modules


def discover_local_modules(entrypoint: Path = ENTRYPOINT) -> set[Path]:
    """Recursively collect the entrypoint and its imported local Python modules."""
    pending = [entrypoint]
    included: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in included:
            continue
        _check_source_path(current)
        if current.suffix != ".py":
            raise PackageError("Only Python source files may be module dependencies")
        included.add(current)
        pending.extend(_local_import_modules(current) - included)

    package_initializers: set[Path] = set()
    for module_path in included:
        parent = module_path.parent
        while parent.parts:
            initializer = parent / "__init__.py"
            if (REPOSITORY_ROOT / initializer).is_file():
                _check_source_path(initializer)
                package_initializers.add(initializer)
            parent = parent.parent
    return included | package_initializers


def _load_and_rewrite_validation_manifest() -> tuple[pd.DataFrame, dict[str, Path]]:
    manifest_path = _check_source_path(VALIDATION_MANIFEST)
    manifest = pd.read_csv(manifest_path)
    required_columns = {"patient_id", "hospital", "file_path"}
    if not required_columns.issubset(manifest.columns):
        raise PackageError("Validation manifest is missing required columns")
    if len(manifest) != EXPECTED_PATIENTS or manifest["patient_id"].nunique() != EXPECTED_PATIENTS:
        raise PackageError("Validation manifest does not contain exactly 6,050 unique patients")

    files: dict[str, Path] = {}
    portable_paths: list[str] = []
    for row_number, source_path in enumerate(manifest["file_path"], start=2):
        candidate = Path(str(source_path))
        if candidate.is_absolute():
            raise PackageError(f"Validation manifest row {row_number} uses an absolute path")
        archive_name = _relative_archive_name(candidate.as_posix())
        if candidate.suffix.lower() != ".psv":
            raise PackageError(f"Validation manifest row {row_number} does not reference a PSV file")
        if candidate.parts[: len(PATIENT_FILE_PREFIX.parts)] != PATIENT_FILE_PREFIX.parts:
            raise PackageError(f"Validation manifest row {row_number} references an unexpected directory")
        if len(candidate.parts) != len(PATIENT_FILE_PREFIX.parts) + 2:
            raise PackageError(f"Validation manifest row {row_number} has an unexpected path depth")
        split_directory = candidate.parts[-2]
        if split_directory not in {"training_setA", "training_setB"}:
            raise PackageError(f"Validation manifest row {row_number} references an unexpected source")
        if archive_name in files:
            raise PackageError("Validation manifest contains duplicate PSV paths")
        files[archive_name] = _check_source_path(candidate)
        portable_paths.append(archive_name)
    rewritten = manifest.copy()
    rewritten["file_path"] = portable_paths
    return rewritten, files


def _render_readme() -> bytes:
    return README.format(python_version=platform.python_version()).encode("utf-8")


def _expected_archive_files() -> tuple[dict[str, Path], pd.DataFrame]:
    included = {
        path.as_posix(): _check_source_path(path)
        for path in discover_local_modules()
    }
    for relative_path in (*REQUIRED_MODEL_PATHS,):
        included[relative_path.as_posix()] = _check_source_path(relative_path)
    manifest, patient_files = _load_and_rewrite_validation_manifest()
    included.update(patient_files)
    return included, manifest


def _write_archive(archive_path: Path, files: dict[str, Path], manifest: pd.DataFrame) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for archive_name, source_path in sorted(files.items()):
            archive.write(source_path, _relative_archive_name(archive_name))
        archive.writestr(
            _relative_archive_name(VALIDATION_MANIFEST.as_posix()),
            manifest.to_csv(index=False).encode("utf-8"),
        )
        archive.writestr(
            _relative_archive_name("requirements-colab.txt"),
            REQUIREMENTS.encode("utf-8"),
        )
        archive.writestr(
            _relative_archive_name("README_COLAB.md"),
            _render_readme(),
        )
        for init_path in (
            "src/__init__.py",
            "src/acquisition/__init__.py",
            "src/data/__init__.py",
            "src/simulation/__init__.py",
            "experiments/__init__.py",
        ):
            if init_path not in files:
                archive.writestr(_relative_archive_name(init_path), "")


def _reject_disallowed_entries(names: Iterable[str]) -> None:
    for name in names:
        normalized = _relative_archive_name(name.rstrip("/"))
        parts = PurePosixPath(normalized).parts
        lowered = [part.lower() for part in parts]
        if any(part in {".git", ".venv", "node_modules"} for part in lowered):
            raise PackageError("Archive includes a forbidden environment or VCS directory")
        if any(part == "test" or part.startswith("test_") for part in lowered):
            raise PackageError("Archive includes a forbidden test artifact")
        if any(part.startswith(".env") or part in {"secrets", "secret"} for part in lowered):
            raise PackageError("Archive includes a forbidden secrets or environment file")
        if "outputs/splits/test.csv" in normalized or "outputs/splits/train.csv" in normalized:
            raise PackageError("Archive includes a non-validation split manifest")
        if normalized.startswith("outputs/models/") and normalized not in {
            path.as_posix() for path in REQUIRED_MODEL_PATHS
        }:
            raise PackageError("Archive includes an unnecessary model artifact")


def validate_archive(
    archive_path: Path,
    expected_patient_names: set[str],
) -> dict[str, int]:
    """Validate archive safety, required models, split paths, and PSV membership."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PackageError("Archive contains duplicate paths")
        _reject_disallowed_entries(names)
        for info in infos:
            _relative_archive_name(info.filename.rstrip("/"))
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PackageError("Archive contains a symbolic link")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise PackageError("Archive integrity check failed")

        member_names = set(names)
        for required_path in REQUIRED_MODEL_PATHS:
            if required_path.as_posix() not in member_names:
                raise PackageError("Archive is missing a required model artifact")
        psv_names = {
            name for name in member_names if name.lower().endswith(".psv")
        }
        if psv_names != expected_patient_names or len(psv_names) != EXPECTED_PATIENTS:
            raise PackageError("Archive PSV membership does not match validation manifest")

        manifest_bytes = archive.read(VALIDATION_MANIFEST.as_posix())
        packaged_manifest = pd.read_csv(io.BytesIO(manifest_bytes))
        if len(packaged_manifest) != EXPECTED_PATIENTS:
            raise PackageError("Packaged validation manifest has an invalid row count")
        packaged_paths = packaged_manifest["file_path"].astype(str).tolist()
        if set(packaged_paths) != expected_patient_names:
            raise PackageError("Packaged validation paths do not resolve to archive members")
        if len(set(packaged_paths)) != EXPECTED_PATIENTS:
            raise PackageError("Packaged validation manifest contains duplicate paths")
        if not ({"patient_id", "hospital", "file_path"} <= set(packaged_manifest.columns)):
            raise PackageError("Packaged validation manifest is missing required columns")
        for required_text in ("requirements-colab.txt", "README_COLAB.md"):
            if required_text not in member_names:
                raise PackageError("Archive is missing Colab setup documentation")
        if "experiments/run_batch_acquisition.py" not in member_names:
            raise PackageError("Archive is missing the experiment entrypoint")
        if not {
            "src/__init__.py",
            "src/acquisition/__init__.py",
            "src/data/__init__.py",
            "src/simulation/__init__.py",
            "experiments/__init__.py",
        }.issubset(member_names):
            raise PackageError("Archive is missing package initialization files")
        return {
            "member_count": len(names),
            "python_module_count": sum(name.endswith(".py") for name in names),
            "patient_file_count": len(psv_names),
            "validation_row_count": len(packaged_manifest),
        }


def package() -> tuple[Path, dict[str, int], str]:
    if OUTPUT_DIR.is_symlink():
        raise PackageError("Refusing to write the package into a symbolic-link directory")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.is_symlink():
        raise PackageError("Refusing to replace a symbolic-link output archive")
    files, manifest = _expected_archive_files()
    expected_patient_names = {
        _relative_archive_name(name) for name in manifest["file_path"].astype(str)
    }
    with tempfile.TemporaryDirectory(prefix="colab-package-") as temporary_directory:
        temporary_archive = Path(temporary_directory) / OUTPUT_ZIP.name
        _write_archive(temporary_archive, files, manifest)
        validation = validate_archive(temporary_archive, expected_patient_names)
        shutil.copyfile(temporary_archive, OUTPUT_ZIP)

    final_validation = validate_archive(OUTPUT_ZIP, expected_patient_names)
    digest = hashlib.sha256()
    with OUTPUT_ZIP.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    return OUTPUT_ZIP, final_validation or validation, checksum


def main() -> None:
    try:
        archive_path, validation, checksum = package()
    except (OSError, ValueError, zipfile.BadZipFile, PackageError) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Colab archive created and validated.")
    print(f"Archive: {archive_path.relative_to(REPOSITORY_ROOT)}")
    print(f"Size: {archive_path.stat().st_size:,} bytes")
    print(f"Validation patient files: {validation['patient_file_count']:,}")
    print(f"Archive members: {validation['member_count']:,}")
    print(f"Python modules: {validation['python_module_count']:,}")
    print(f"SHA256: {checksum}")
    print("Colab package output directory is ignored by Git.")


if __name__ == "__main__":
    main()
