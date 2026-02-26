from app import config
from app.github_service import GitHubService
from app.models import RepoTreeItem


def test_prefilter_skips_binary_and_vendor_paths():
    service = GitHubService()
    tree = [
        RepoTreeItem(path="README.md", size=100, type="blob"),
        RepoTreeItem(path="images/logo.png", size=120, type="blob"),
        RepoTreeItem(path="vendor/lib/a.py", size=100, type="blob"),
        RepoTreeItem(path="node_modules/pkg/index.js", size=100, type="blob"),
        RepoTreeItem(path="src/main.py", size=100, type="blob"),
    ]
    filtered = service.prefilter_tree(tree)
    paths = {item.path for item in filtered}

    assert "README.md" in paths
    assert "src/main.py" in paths
    assert "images/logo.png" not in paths
    assert "vendor/lib/a.py" not in paths
    assert "node_modules/pkg/index.js" not in paths


def test_fallback_selection_enforces_max_selected_files_and_keeps_readme():
    service = GitHubService()
    files = [RepoTreeItem(path="README.md", size=120, type="blob")]
    files.extend(
        RepoTreeItem(path=f"src/module_{i}.py", size=100 + i, type="blob")
        for i in range(config.MAX_SELECTED_FILES + 5)
    )
    selected = service.fallback_file_selection(files)

    assert len(selected) <= config.MAX_SELECTED_FILES
    assert "README.md" in selected
