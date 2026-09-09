"""Exercise multiple previews, failed updates, pruning, and change detection."""

from pathlib import Path
import tempfile
import unittest

from previews import (
    STATE_FILE,
    assemble,
    load_previews,
    make_plan,
    write_json,
)


class PreviewPublicationTests(unittest.TestCase):
    """Model publication cycles using ordinary files, without network calls."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "published"
        self.site.mkdir()
        self.artifacts = self.root / "artifacts"
        self.previews = {"main": "main", "pxp": "codex/pxp"}
        self.heads = {"main": "a" * 40, "codex/pxp": "b" * 40}

    def build(self, slug, content):
        artifact = self.artifacts / f"preview-{slug}"
        (artifact / "site").mkdir(parents=True, exist_ok=True)
        (artifact / "site/index.html").write_text(content)
        write_json(artifact / "result.json", {"status": "success"})

    def initial_publication(self):
        self.build("main", "main v1")
        self.build("pxp", "pxp v1")
        plan = make_plan(self.previews, self.heads, {}, "recipe")
        return assemble(plan, self.site, self.artifacts)

    def test_two_branches_remain_online_and_unchanged_commits_skip_build(self):
        state = self.initial_publication()
        self.assertEqual((self.site / "main/index.html").read_text(), "main v1")
        self.assertEqual((self.site / "pxp/index.html").read_text(), "pxp v1")
        plan = make_plan(self.previews, self.heads, state, "recipe")
        self.assertFalse(plan["publish"])
        self.assertFalse(any(entry["build"] for entry in plan["entries"]))

    def test_successful_update_replaces_only_its_directory(self):
        state = self.initial_publication()
        (self.site / "pxp/stale.html").write_text("obsolete")
        self.heads["codex/pxp"] = "c" * 40
        self.build("pxp", "pxp v2")
        plan = make_plan(self.previews, self.heads, state, "recipe")
        updated = assemble(plan, self.site, self.artifacts)
        self.assertEqual((self.site / "pxp/index.html").read_text(), "pxp v2")
        self.assertFalse((self.site / "pxp/stale.html").exists())
        self.assertEqual((self.site / "main/index.html").read_text(), "main v1")
        self.assertEqual(updated["pxp"]["sha"], "c" * 40)

    def test_failed_update_preserves_previous_site_and_commit(self):
        state = self.initial_publication()
        self.heads["codex/pxp"] = "c" * 40
        write_json(self.artifacts / "preview-pxp/result.json", {"status": "failure"})
        plan = make_plan(self.previews, self.heads, state, "recipe")
        updated = assemble(plan, self.site, self.artifacts)
        self.assertEqual((self.site / "pxp/index.html").read_text(), "pxp v1")
        self.assertEqual(updated["pxp"]["sha"], "b" * 40)
        self.assertEqual(updated["pxp"]["status"], "failed")
        self.assertIn(
            "previous preview retained", (self.site / "index.html").read_text()
        )
        self.assertFalse(
            make_plan(self.previews, self.heads, updated, "recipe")["publish"]
        )
        retry = make_plan(self.previews, self.heads, updated, "recipe", force=True)
        self.assertTrue(all(entry["build"] for entry in retry["entries"]))
        self.heads["codex/pxp"] = "d" * 40
        retry = make_plan(self.previews, self.heads, updated, "recipe")
        self.assertEqual([e["slug"] for e in retry["entries"] if e["build"]], ["pxp"])

    def test_missing_artifact_does_not_replace_a_published_preview(self):
        state = self.initial_publication()
        self.heads["main"] = "c" * 40
        plan = make_plan(self.previews, self.heads, state, "recipe")
        updated = assemble(plan, self.site, self.root / "no-artifacts")
        self.assertEqual(updated["main"]["status"], "failed")
        self.assertEqual((self.site / "main/index.html").read_text(), "main v1")

    def test_removed_remote_branch_retains_site_until_removed_from_config(self):
        state = self.initial_publication()
        del self.heads["codex/pxp"]
        plan = make_plan(self.previews, self.heads, state, "recipe")
        state = assemble(plan, self.site, self.artifacts)
        self.assertEqual(state["pxp"]["status"], "missing")
        self.assertTrue((self.site / "pxp/index.html").exists())
        del self.previews["pxp"]
        plan = make_plan(self.previews, self.heads, state, "recipe")
        state = assemble(plan, self.site, self.artifacts)
        self.assertNotIn("pxp", state)
        self.assertFalse((self.site / "pxp").exists())
        self.assertTrue((self.site / "main/index.html").exists())

    def test_new_branch_builds_without_rebuilding_existing_branches(self):
        state = self.initial_publication()
        self.previews["new"] = "docs/new"
        self.heads["docs/new"] = "c" * 40
        plan = make_plan(self.previews, self.heads, state, "recipe")
        self.assertEqual([e["slug"] for e in plan["entries"] if e["build"]], ["new"])
        changed_recipe = make_plan(self.previews, self.heads, state, "new recipe")
        self.assertTrue(all(entry["build"] for entry in changed_recipe["entries"]))

    def test_missing_unpublished_branch_can_be_removed(self):
        plan = make_plan({"missing": "missing"}, {}, {}, "recipe")
        state = assemble(plan, self.site, self.artifacts)
        self.assertNotIn('href="missing/"', (self.site / "index.html").read_text())
        plan = make_plan({}, {}, state, "recipe")
        self.assertEqual(assemble(plan, self.site, self.artifacts), {})

    def test_symlinks_are_rejected_before_replacing_published_files(self):
        state = self.initial_publication()
        self.heads["codex/pxp"] = "c" * 40
        (self.artifacts / "preview-pxp/site/link").symlink_to(self.site / STATE_FILE)
        plan = make_plan(self.previews, self.heads, state, "recipe")
        with self.assertRaises(ValueError):
            assemble(plan, self.site, self.artifacts)
        self.assertEqual((self.site / "pxp/index.html").read_text(), "pxp v1")

    def test_unsafe_directory_names_and_control_branches_are_rejected(self):
        config = self.root / "previews.toml"
        for slug in ("../main", ".git", "a/b", "A", "x" * 65):
            with self.subTest(slug=slug):
                config.write_text(f'[previews]\n"{slug}" = "main"\n')
                with self.assertRaises(ValueError):
                    load_previews(config)
        config.write_text('[previews]\nmain = "gh-pages"\n')
        with self.assertRaises(ValueError):
            load_previews(config)


if __name__ == "__main__":
    unittest.main()
