"""Plan and assemble the fork's independent documentation previews."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import quote

REPOSITORY = "hmyuuu/fatqat"
SITE_URL = "https://hmyuuu.github.io/fatqat/"
REMOTE = f"https://github.com/{REPOSITORY}.git"
STATE_FILE = ".previews.json"
ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path, default=None):
    """Read optional persisted state, without hiding invalid JSON."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, value) -> None:
    """Write deterministic JSON so unchanged state does not create commits."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_previews(path: Path) -> dict[str, str]:
    """Validate directory names before using them in paths or artifact names."""
    previews = tomllib.loads(path.read_text(encoding="utf-8"))["previews"]
    if len(previews) > 256:
        raise ValueError("GitHub Actions supports at most 256 matrix builds")
    for slug, branch in previews.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", slug):
            raise ValueError(f"Invalid preview directory: {slug!r}")
        if (
            not isinstance(branch, str)
            or not branch
            or branch
            in {
                "gh-pages",
                "codex/docs-hosting",
            }
        ):
            raise ValueError(f"Invalid source branch: {branch!r}")
    return previews


def git(*args: str, cwd: Path | None = None) -> str:
    """Run Git without a shell, propagating failures."""
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def remote_heads() -> dict[str, str]:
    """Resolve all branches in a single remote request."""
    return {
        ref.removeprefix("refs/heads/"): sha
        for sha, ref in (
            line.split() for line in git("ls-remote", "--heads", REMOTE).splitlines()
        )
    }


def restore_site(destination: Path, heads: dict[str, str]) -> None:
    """Restore successful publications, or initialize the first site's branch."""
    if "gh-pages" in heads:
        git("clone", "--depth=1", "--branch=gh-pages", REMOTE, str(destination))
    else:
        destination.mkdir(parents=True)
        git("init", "--initial-branch=gh-pages", cwd=destination)
        git("remote", "add", "origin", REMOTE, cwd=destination)


def make_plan(previews, heads, state, recipe, *, force=False):
    """Rebuild changed commits or recipes; retry failed commits only on request."""
    entries = []
    for slug, branch in previews.items():
        sha = heads.get(branch)
        url = f"{SITE_URL}{slug}/"
        fingerprint = hashlib.sha256(
            json.dumps([branch, sha, recipe, url]).encode()
        ).hexdigest()
        previous = state.get(slug, {})
        entries.append(
            {
                "slug": slug,
                "branch": branch,
                "sha": sha,
                "url": url,
                "fingerprint": fingerprint,
                "build": bool(sha)
                and (force or previous.get("attempt") != fingerprint),
            }
        )
    publish = set(previews) != set(state) or any(
        entry["build"]
        or state.get(entry["slug"], {}).get("attempt") != entry["fingerprint"]
        for entry in entries
    )
    return {"entries": entries, "publish": publish}


def render_index(state) -> str:
    """Render a small index that distinguishes current and retained previews."""
    rows = []
    for slug, entry in sorted(state.items()):
        branch = html.escape(entry["branch"])
        source = (
            f"https://github.com/{REPOSITORY}/tree/{quote(entry['branch'], safe='')}"
        )
        preview = (
            f'<a href="{slug}/">Open preview</a>'
            if entry.get("sha")
            else "Not published yet"
        )
        status = {
            "ready": "Up to date",
            "failed": (
                "Build failed; previous preview retained"
                if entry.get("sha")
                else "Build failed"
            ),
            "missing": (
                "Branch missing; previous preview retained"
                if entry.get("sha")
                else "Branch missing"
            ),
        }[entry["status"]]
        commit = entry.get("sha", "")
        revision = (
            f'<a href="https://github.com/{REPOSITORY}/commit/{commit}">{commit[:7]}</a>'
            if commit
            else "—"
        )
        rows.append(
            f'<tr><td><strong>{slug}</strong><br><a href="{source}">{branch}</a></td>'
            f"<td>{preview}</td><td>{revision}</td><td>{status}</td></tr>"
        )
    return (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow"><title>FatQat branch previews</title>
<style>
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { max-width: 1100px; margin: 4rem auto; padding: 0 1.5rem; line-height: 1.6; }
h1 { line-height: 1.2; } a { color: light-dark(#075ca8, #8cc8ff); }
.table { overflow-x: auto; } table { width: 100%; border-collapse: collapse; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #8886; }
td:first-child { overflow-wrap: anywhere; } footer { margin-top: 2rem; }
</style><h1>FatQat branch previews</h1>
<p>Development documentation from hmyuuu/fatqat. Each preview follows its own branch.</p>
<div class="table"><table><thead><tr><th>Branch</th><th>Documentation</th><th>Published commit</th><th>Status</th></tr></thead>
<tbody>"""
        + "\n".join(rows)
        + f"""</tbody></table></div>
<footer><a href="https://github.com/{REPOSITORY}/actions/workflows/docs-previews.yml">Build history and refresh</a>
 · <a href="https://github.com/{REPOSITORY}/blob/codex/docs-hosting/hosting/previews.toml">Manage previews</a>
 · <a href="https://fatqat.readthedocs.io/en/latest/">Official documentation</a></footer></html>
"""
    )


def assemble(plan, site: Path, artifacts: Path) -> dict:
    """Replace successful builds only, preserving the last good failed previews."""
    state = read_json(site / STATE_FILE, {})
    active = {entry["slug"] for entry in plan["entries"]}
    for slug in set(state) - active:
        if (site / slug).exists():
            shutil.rmtree(site / slug)
        del state[slug]
    for entry in plan["entries"]:
        slug = entry["slug"]
        previous = state.get(slug, {})
        if not entry["build"] and entry["sha"]:
            continue
        current = dict(previous, branch=entry["branch"], attempt=entry["fingerprint"])
        current["status"] = "failed" if entry["sha"] else "missing"
        artifact = artifacts / f"preview-{slug}"
        result = read_json(artifact / "result.json", {})
        built = artifact / "site"
        if (
            entry["build"]
            and result.get("status") == "success"
            and (built / "index.html").is_file()
        ):
            # Pages artifacts must contain ordinary files and directories only.
            for path in built.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError(f"Unsupported file in preview: {path}")
            if (site / slug).exists():
                shutil.rmtree(site / slug)
            shutil.copytree(built, site / slug)
            current.update(status="ready", sha=entry["sha"])
        state[slug] = current
    write_json(site / STATE_FILE, state)
    (site / ".nojekyll").touch()
    (site / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    (site / "index.html").write_text(render_index(state), encoding="utf-8")
    return state


def main() -> None:
    """Run the internal planning or publication step used by GitHub Actions."""
    temporary = Path(os.environ["RUNNER_TEMP"])
    site = temporary / "published"
    heads = remote_heads()
    restore_site(site, heads)
    if sys.argv[1] == "plan":
        recipe = hashlib.sha256(
            (ROOT / ".github/workflows/docs-previews.yml").read_bytes()
            + Path(__file__).read_bytes()
        ).hexdigest()
        plan = make_plan(
            load_previews(ROOT / "hosting/previews.toml"),
            heads,
            read_json(site / STATE_FILE, {}),
            recipe,
            force=os.environ.get("FORCE_REBUILD") == "true",
        )
        plan["publish"] |= not (site / "index.html").exists()
        write_json(temporary / "plan.json", plan)
        matrix = [entry for entry in plan["entries"] if entry["build"]]
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"matrix={json.dumps(matrix)}\n")
            output.write(f"has_builds={str(bool(matrix)).lower()}\n")
            output.write(f"publish={str(plan['publish']).lower()}\n")
    elif sys.argv[1] == "publish":
        state = assemble(
            read_json(temporary / "plan/plan.json"), site, temporary / "artifacts"
        )
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"[All previews]({SITE_URL})\n\n")
            for slug, entry in state.items():
                summary.write(f"- [{slug}]({SITE_URL}{slug}/): {entry['status']}\n")
    else:
        raise ValueError("Expected plan or publish")


if __name__ == "__main__":
    main()
