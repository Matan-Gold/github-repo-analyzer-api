from evaluation import extract_declared_dependencies, infer_languages_from_extensions, validate_technologies


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
