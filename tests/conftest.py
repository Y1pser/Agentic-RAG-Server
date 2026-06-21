"""Shared pytest fixtures for all test layers."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir():
    """Return the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_doc_path():
    """Return path to a sample document for ingestion tests."""
    return Path(__file__).parent / "fixtures" / "sample_documents" / "hello.txt"
