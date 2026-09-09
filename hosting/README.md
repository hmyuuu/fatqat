# Fork documentation previews

This infrastructure belongs to `hmyuuu/fatqat`, on `codex/docs-hosting` only.
The preview index is <https://hmyuuu.github.io/fatqat/>. The fork's default
branch is the hosting branch so scheduled and manual refreshes run centrally.
Feature branches and upstream pull requests do not need these files.

## Add or remove a preview

Edit `hosting/previews.toml` on `codex/docs-hosting`:

```toml
[previews]
main = "main"
pxp = "codex/pxp-incremental-evolution"
homepage = "docs/homepage-redesign"
```

Each value must name an existing branch in this fork. Keys are unique URL
directories, using lowercase letters, numbers, and hyphens, up to 64 characters.
For example, `homepage` publishes at `/fatqat/homepage/`. Slashes in source
branch names are supported. Add as many previews as needed, up to GitHub's
256-job matrix limit; four branches build concurrently.

Push the configuration change to `codex/docs-hosting` to refresh the site.
Removing an entry removes its published directory on the next successful
deployment. A missing source branch retains any previously published preview
and displays its status on the index until its entry is removed.

## Refresh previews

The workflow checks branch heads at minutes 7, 22, 37, and 52 of every hour.
Only changed commits or build recipes are rebuilt. Adding a preview does not
rebuild existing previews. A push to a source branch is picked up by the next
scheduled check; it does not directly trigger this fork-only workflow.
GitHub may delay schedules, and disables scheduled runs after 60 days without
repository activity. Re-enable the workflow in Actions if that occurs.

For an immediate check, choose **Actions → Documentation previews (fork only)
→ Run workflow**, using the `codex/docs-hosting` branch. Or run:

```sh
gh workflow run docs-previews.yml --repo hmyuuu/fatqat --ref codex/docs-hosting
```

To retry a failed build or rebuild unchanged commits, select **force**, or run:

```sh
gh workflow run docs-previews.yml --repo hmyuuu/fatqat --ref codex/docs-hosting -f force=true
```

Builds use each branch's project and pinned docs requirements with Python 3.12.
They run `mkdocs build --strict`, including the branch's tutorial execution and
content validation. A temporary MkDocs configuration supplies the preview URL
and fork source links; the source branch's configuration is not modified.

Failed builds leave the previous successful preview online. The index shows
the failure and the commit actually published. Failed commits are retried when
the branch changes, the hosting build recipe changes, or a forced refresh is
requested. Consult the failed build job for the original error.

## Publication and isolation

Build jobs have read-only repository permissions and no saved Git credentials.
One separate publishing job assembles all available previews, uploads the
complete Pages artifact, and deploys it. Only after deployment succeeds does
it save the generated tree and commit metadata to `gh-pages`. The workflow is
serialized so concurrent refreshes cannot overwrite each other.

Pages must use **GitHub Actions** as its source. The `github-pages` deployment
environment allows only `codex/docs-hosting`. Keep that branch as the fork's
default branch, and keep this workflow's repository guard in place.

The `gh-pages` branch contains generated files only; never merge it or
`codex/docs-hosting` into upstream feature branches. Start contributions from
`upstream/main` rather than the fork's default branch. Existing source branches
continue to work normally.

## Validate hosting changes

```sh
python -m unittest discover -s hosting -p 'test_*.py'
actionlint .github/workflows/docs-previews.yml
git diff --check
```

The tests cover simultaneous previews, successful and failed updates, missing
artifacts and branches, explicit removal, and change detection. Every Actions
refresh runs these checks on the Python publication logic before planning builds.
