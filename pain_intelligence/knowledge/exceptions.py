"""Custom exceptions for pipeline knowledge management."""

from __future__ import annotations

from pathlib import Path


class StaleAssetError(Exception):
    """Raised when a downstream stage detects an asset from an older pipeline run."""

    def __init__(self, asset_path: str | Path, expected_run_id: str, actual_run_id: str) -> None:
        self.asset_path = str(asset_path)
        self.expected_run_id = expected_run_id
        self.actual_run_id = actual_run_id
        super().__init__(
            f"StaleAssetError: {asset_path} belongs to a previous pipeline run.\n"
            f"  Expected run_id: {expected_run_id}\n"
            f"  Actual run_id:   {actual_run_id}\n"
            "  Re-run the upstream pipeline stage."
        )


class MissingAssetError(Exception):
    """Raised when a required upstream asset does not exist."""

    def __init__(self, asset_path: str | Path) -> None:
        self.asset_path = str(asset_path)
        super().__init__(f"MissingAssetError: required asset not found: {asset_path}")


class SchemaMismatchError(Exception):
    """Raised when an asset's schema does not match expectations."""

    def __init__(self, asset_path: str | Path, detail: str) -> None:
        self.asset_path = str(asset_path)
        super().__init__(f"SchemaMismatchError: {asset_path} — {detail}")


class ChecksumMismatchError(Exception):
    """Raised when an asset's checksum does not match the manifest."""

    def __init__(self, asset_path: str | Path, expected: str, actual: str) -> None:
        self.asset_path = str(asset_path)
        super().__init__(
            f"ChecksumMismatchError: {asset_path}\n"
            f"  Expected checksum: {expected}\n"
            f"  Actual checksum:   {actual}"
        )
