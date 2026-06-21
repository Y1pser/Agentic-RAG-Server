"""Verify shared pytest fixtures from conftest.py work correctly."""


def test_project_root_fixture(project_root):
    """project_root should point to the actual project directory."""
    assert project_root.exists()
    assert (project_root / "main.py").exists()
    assert (project_root / "pyproject.toml").exists()


def test_fixtures_dir_fixture(fixtures_dir):
    """fixtures_dir should point to tests/fixtures/."""
    assert fixtures_dir.exists()
    assert fixtures_dir.is_dir()


def test_sample_doc_path_fixture(sample_doc_path):
    """sample_doc_path should point to the hello.txt sample document."""
    assert sample_doc_path.exists()
    assert sample_doc_path.is_file()
    content = sample_doc_path.read_text()
    assert "RAG" in content
    assert "test document" in content
