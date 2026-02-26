"""Deterministic evidence validation utilities."""

from __future__ import annotations

import json
import re


SAFE_TECH_WHITELIST = {"python", "javascript", "java", "go", "docker", "rest", "api"}

GENERIC_STRUCTURE_TERMS = {
    "project",
    "repository",
    "codebase",
    "source",
    "module",
    "component",
    "service",
    "api",
    "docs",
    "documentation",
    "tests",
    "config",
    "configuration",
}

COMMON_FILE_EXTENSIONS = {
    "py",
    "md",
    "rst",
    "txt",
    "toml",
    "json",
    "yml",
    "yaml",
    "ini",
    "cfg",
    "xml",
    "go",
    "rs",
    "js",
    "ts",
    "tsx",
    "jsx",
    "java",
    "sh",
}


def _normalize_name(value: str) -> str:
    cleaned = value.strip().strip("\"'`")
    cleaned = cleaned.split("/")[-1]
    cleaned = re.split(r"[<>=!~\s@:\[\]()]", cleaned, maxsplit=1)[0]
    return cleaned.lower()


def _extract_pyproject_fallback(content: str) -> set[str]:
    deps: set[str] = set()

    # Parse "dependencies = [ ... ]" under [project] with a conservative regex.
    project_block = re.search(r"\[project\](?P<body>.*?)(?:\n\[|$)", content, flags=re.S | re.I)
    if project_block:
        dep_list = re.search(r"dependencies\s*=\s*\[(?P<deps>.*?)\]", project_block.group("body"), flags=re.S | re.I)
        if dep_list:
            for dep in re.findall(r"['\"]([^'\"]+)['\"]", dep_list.group("deps")):
                name = _normalize_name(dep)
                if name:
                    deps.add(name)

    # Parse [tool.poetry.dependencies] keys.
    poetry_block = re.search(r"\[tool\.poetry\.dependencies\](?P<body>.*?)(?:\n\[|$)", content, flags=re.S | re.I)
    if poetry_block:
        for raw_line in poetry_block.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", maxsplit=1)[0].strip()
            name = _normalize_name(key)
            if name and name != "python":
                deps.add(name)

    return deps


def _looks_like_file_token(token: str, repo_paths: list[str]) -> bool:
    cleaned = token.strip(".,:;()[]{}")
    if not cleaned:
        return False
    if "/" in cleaned:
        return True
    if "." not in cleaned:
        return False

    known_exts = {path.rsplit(".", 1)[1].lower() for path in repo_paths if "." in path}
    known_exts |= COMMON_FILE_EXTENSIONS
    stem, ext = cleaned.rsplit(".", 1)
    return len(stem) > 1 and ext.lower() in known_exts


def _soften_unverified_path_claim(text: str, path_refs: list[str]) -> str:
    # Remove ungrounded file-path tokens while keeping any generic structural wording.
    softened = text
    for ref in sorted(set(path_refs), key=len, reverse=True):
        softened = re.sub(rf"\b{re.escape(ref)}\b", "", softened)
    softened = re.sub(r"\s{2,}", " ", softened)
    softened = softened.strip(" ,;:-")
    return softened.strip()


def extract_declared_dependencies(files_dict: dict[str, str]) -> set[str]:
    deps: set[str] = set()

    for path, content in files_dict.items():
        lower_path = path.lower()

        if lower_path.endswith("requirements.txt"):
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                name = _normalize_name(line)
                if name:
                    deps.add(name)

        elif lower_path.endswith("pyproject.toml"):
            try:
                try:
                    import tomllib as _tomllib  # type: ignore[attr-defined]
                except ModuleNotFoundError:
                    import tomli as _tomllib  # type: ignore[import-not-found]

                parsed = _tomllib.loads(content)
                project = parsed.get("project", {})
                for dep in project.get("dependencies", []):
                    name = _normalize_name(str(dep))
                    if name:
                        deps.add(name)
                poetry = parsed.get("tool", {}).get("poetry", {}).get("dependencies", {})
                if isinstance(poetry, dict):
                    for dep in poetry.keys():
                        name = _normalize_name(str(dep))
                        if name and name != "python":
                            deps.add(name)
            except Exception:
                deps.update(_extract_pyproject_fallback(content))

        elif lower_path.endswith("package.json"):
            try:
                parsed = json.loads(content)
                for section in ("dependencies", "devDependencies", "peerDependencies"):
                    values = parsed.get(section, {})
                    if isinstance(values, dict):
                        for dep in values.keys():
                            name = _normalize_name(str(dep))
                            if name:
                                deps.add(name)
            except Exception:
                continue

        elif lower_path.endswith("go.mod"):
            in_require_block = False
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                if line.startswith("require ("):
                    in_require_block = True
                    continue
                if in_require_block and line == ")":
                    in_require_block = False
                    continue
                if line.startswith("require "):
                    dep = line[len("require ") :].split()[0]
                    name = _normalize_name(dep)
                    if name:
                        deps.add(name)
                elif in_require_block:
                    dep = line.split()[0]
                    name = _normalize_name(dep)
                    if name:
                        deps.add(name)

        elif lower_path.endswith("pom.xml"):
            matches = re.findall(r"<artifactId>([^<]+)</artifactId>", content, flags=re.IGNORECASE)
            for dep in matches:
                name = _normalize_name(dep)
                if name:
                    deps.add(name)

    return deps


def infer_languages_from_extensions(file_paths: list[str]) -> set[str]:
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "javascript",
        ".tsx": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c#",
        ".cpp": "c++",
        ".c": "c",
        ".kt": "kotlin",
        ".swift": "swift",
        ".sh": "shell",
    }
    langs: set[str] = set()
    for path in file_paths:
        dot = path.rfind(".")
        if dot < 0:
            continue
        ext = path[dot:].lower()
        lang = ext_map.get(ext)
        if lang:
            langs.add(lang)
    return langs


def validate_technologies(
    llm_technologies: list[str],
    dependency_set: set[str],
    inferred_langs: set[str],
) -> list[str]:
    validated: list[str] = []
    seen: set[str] = set()

    for raw in llm_technologies:
        tech = raw.strip()
        key = _normalize_name(tech)
        if not key or key in seen:
            continue
        if key in dependency_set or key in inferred_langs or key in SAFE_TECH_WHITELIST:
            validated.append(tech)
            seen.add(key)
    return validated


def ground_structure_points(structure_points: list[str], repo_paths: list[str]) -> list[str]:
    repo_paths_lower = {path.lower() for path in repo_paths}
    grounded: list[str] = []
    for point in structure_points:
        text = point.strip()
        if not text:
            continue
        tokens = re.findall(r"[A-Za-z0-9_.\-/]+", text)
        path_refs = [tok.strip(".,:;()[]{}") for tok in tokens if _looks_like_file_token(tok, repo_paths)]

        if not path_refs:
            grounded.append(text)
            continue

        has_known_path = any(tok.lower().strip(".,:;") in repo_paths_lower for tok in path_refs)
        if has_known_path:
            grounded.append(text)
            continue

        # If no path is grounded, keep only generic structural wording and remove bad path claims.
        lowered = text.lower()
        if any(term in lowered for term in GENERIC_STRUCTURE_TERMS):
            softened = _soften_unverified_path_claim(text, path_refs)
            if softened:
                grounded.append(f"{softened} (generalized from available repository evidence).")
            else:
                grounded.append("Repository structure includes generic source, configuration, and docs areas.")
    return grounded


def compute_confidence_score(
    validated_tech: list[str],
    original_tech: list[str],
    grounded_structure: list[str],
    original_structure: list[str],
) -> float:
    tech_ratio = 1.0 if not original_tech else len(validated_tech) / max(1, len(original_tech))
    structure_ratio = 1.0 if not original_structure else len(grounded_structure) / max(1, len(original_structure))
    score = (tech_ratio + structure_ratio) / 2.0
    return max(0.0, min(1.0, round(score, 4)))
