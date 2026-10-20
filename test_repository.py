from pathlib import Path

def test_expected_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").is_file()
    assert (root / "requirements.txt").is_file()
    assert (root / "README.md").is_file()
