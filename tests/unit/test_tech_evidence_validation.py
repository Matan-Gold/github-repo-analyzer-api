from evaluation import (
    extract_declared_dependencies,
    ground_structure_points,
    infer_languages_from_extensions,
    validate_technologies,
)


def test_validate_technologies_drops_unsupported_items():
    files_dict = {
        "requirements.txt": "flask==3.0.0\nrequests==2.32.5\n",
        "src/app.py": "import flask\n",
    }
    deps = extract_declared_dependencies(files_dict)
    langs = infer_languages_from_extensions(list(files_dict.keys()))

    result = validate_technologies(["Flask", "Django", "Requests"], deps, langs)
    lowered = {item.lower() for item in result}

    assert "flask" in lowered
    assert "requests" in lowered
    assert "django" not in lowered


def test_ground_structure_points_softens_unknown_file_references():
    repo_paths = ["README.md", "src/app.py"]
    points = ["requirements.txt: dependency and build configuration."]

    result = ground_structure_points(points, repo_paths)

    assert len(result) == 1
    assert "requirements.txt" not in result[0].lower()
    assert "generalized from available repository evidence" in result[0].lower()


def test_ground_structure_points_keeps_known_file_references():
    repo_paths = ["requirements.txt", "src/app.py"]
    points = ["requirements.txt: dependency and build configuration."]

    result = ground_structure_points(points, repo_paths)

    assert result == points
