"""Update checker logic. Tests never call the live GitHub API."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.update_checker import (  # noqa: E402
    GITHUB_API_LATEST,
    GITHUB_RELEASES_PAGE,
    UpdateStatus,
    asset_sha256,
    can_self_update,
    check_for_update,
    dmg_filename,
    evaluate_release,
    is_newer,
    parse_release_assets,
    parse_sha256sums,
    parse_update_manifest,
    parse_version,
    safe_release_url,
    should_auto_check,
    update_checks_allowed,
    update_is_required,
)


def _payload(**overrides):
    data = {
        "tag_name": "v0.2.0",
        "draft": False,
        "prerelease": False,
        "html_url": "https://github.com/haogesong2011-collab/tracklab/releases/tag/v0.2.0",
        "body": "修复更新检查",
        "published_at": "2026-09-01T12:00:00Z",
        "assets": [
            {
                "name": "TrackLab-arm64.dmg",
                "browser_download_url": (
                    "https://github.com/haogesong2011-collab/tracklab/"
                    "releases/download/v0.2.0/TrackLab-arm64.dmg"
                ),
                "size": 1000,
                "digest": "sha256:" + ("ab" * 32),
            },
            {
                "name": "SHA256SUMS.txt",
                "browser_download_url": (
                    "https://github.com/haogesong2011-collab/tracklab/"
                    "releases/download/v0.2.0/SHA256SUMS.txt"
                ),
                "size": 80,
            },
        ],
    }
    data.update(overrides)
    return data


class UpdateCheckerTests(unittest.TestCase):
    def test_parse_version_requires_three_integers(self) -> None:
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(parse_version("0.1.0"), (0, 1, 0))
        self.assertIsNone(parse_version("v1.2"))
        self.assertIsNone(parse_version("1.2.3.4"))
        self.assertIsNone(parse_version("v1.2.x"))
        self.assertIsNone(parse_version(""))

    def test_semantic_compare_not_lexicographic(self) -> None:
        self.assertTrue(is_newer("v0.10.0", "0.9.0"))
        self.assertFalse(is_newer("v0.9.0", "0.10.0"))
        self.assertFalse(is_newer("v0.1.0", "0.1.0"))
        self.assertFalse(is_newer("not-a-version", "0.1.0"))

    def test_evaluate_available_latest_skip_and_invalid(self) -> None:
        available = evaluate_release(_payload(), "0.1.0")
        self.assertEqual(available.status, UpdateStatus.AVAILABLE)
        self.assertEqual(available.latest, "v0.2.0")
        self.assertEqual(available.published_at, "2026-09-01")
        self.assertIn("github.com", available.download_url)

        latest = evaluate_release(_payload(tag_name="v0.1.0"), "0.1.0")
        self.assertEqual(latest.status, UpdateStatus.LATEST)

        skipped = evaluate_release(_payload(), "0.1.0", skipped="v0.2.0")
        self.assertEqual(skipped.status, UpdateStatus.SKIPPED)

        manual = evaluate_release(
            _payload(), "0.1.0", skipped="v0.2.0", honor_skip=False
        )
        self.assertEqual(manual.status, UpdateStatus.AVAILABLE)

        invalid = evaluate_release(_payload(tag_name="nightly"), "0.1.0")
        self.assertEqual(invalid.status, UpdateStatus.INVALID)

        draft = evaluate_release(_payload(draft=True), "0.1.0")
        self.assertEqual(draft.status, UpdateStatus.NO_RELEASE)

    def test_safe_release_url_https_github_only(self) -> None:
        self.assertIsNotNone(
            safe_release_url(
                "https://github.com/haogesong2011-collab/tracklab/releases/tag/v0.2.0"
            )
        )
        self.assertIsNone(safe_release_url("http://github.com/haogesong2011-collab/tracklab"))
        self.assertIsNone(safe_release_url("https://evil.example/download"))

    def test_throttle_is_twenty_four_hours(self) -> None:
        self.assertTrue(should_auto_check(0, 10.0))
        self.assertFalse(should_auto_check(100.0, 100.0 + 23 * 3600))
        self.assertTrue(should_auto_check(100.0, 100.0 + 24 * 3600))

    def test_fetch_status_mapping_without_network(self) -> None:
        def ok(url, headers, timeout):
            self.assertEqual(url, GITHUB_API_LATEST)
            self.assertIn("TrackLab/", headers["User-Agent"])
            self.assertEqual(timeout, 8.0)
            body = json.dumps(_payload()).encode()
            return 200, {}, body

        info = check_for_update("0.1.0", transport=ok)
        self.assertEqual(info.status, UpdateStatus.AVAILABLE)
        self.assertEqual(info.notes, "修复更新检查")

        info = check_for_update(
            "0.1.0",
            transport=lambda *_a: (404, {}, b""),
        )
        self.assertEqual(info.status, UpdateStatus.NO_RELEASE)

        info = check_for_update(
            "0.1.0",
            transport=lambda *_a: (429, {}, b""),
        )
        self.assertEqual(info.status, UpdateStatus.RATE_LIMITED)

        info = check_for_update(
            "0.1.0",
            transport=lambda *_a: (200, {}, b"not-json"),
        )
        self.assertEqual(info.status, UpdateStatus.INVALID)

        def boom(*_a):
            raise TimeoutError("timed out")

        info = check_for_update("0.1.0", transport=boom)
        self.assertEqual(info.status, UpdateStatus.TIMEOUT)

        def offline(*_a):
            raise URLError("network down")

        info = check_for_update("0.1.0", transport=offline)
        self.assertEqual(info.status, UpdateStatus.OFFLINE)

    def test_download_url_falls_back_to_releases_page(self) -> None:
        info = evaluate_release(_payload(html_url="http://insecure.example"), "0.1.0")
        self.assertEqual(info.download_url, GITHUB_RELEASES_PAGE)

    def test_dmg_assets_and_can_self_update(self) -> None:
        self.assertEqual(dmg_filename("arm64"), "TrackLab-arm64.dmg")
        self.assertEqual(dmg_filename("x86_64"), "TrackLab-x86_64.dmg")
        self.assertIsNone(dmg_filename("riscv64"))
        info = evaluate_release(_payload(), "0.1.0")
        self.assertEqual(info.installer_for("arm64").name, "TrackLab-arm64.dmg")
        self.assertIsNone(info.installer_for("x86_64"))
        self.assertEqual(asset_sha256(info.installer_for("arm64")), "ab" * 32)
        self.assertEqual(len(parse_release_assets(_payload())), 2)
        sums = parse_sha256sums(("aa" * 32) + "  TrackLab-arm64.dmg\n# skip\n")
        self.assertEqual(sums["TrackLab-arm64.dmg"], "aa" * 32)
        self.assertEqual(parse_sha256sums("not-a-hash  file.dmg"), {})
        with tempfile.TemporaryDirectory() as raw:
            bundle = Path(raw) / "TrackLab.app"
            bundle.mkdir()
            self.assertTrue(
                can_self_update(
                    info,
                    frozen=True,
                    system="darwin",
                    machine="arm64",
                    bundle=bundle,
                )
            )
            self.assertFalse(
                can_self_update(
                    info,
                    frozen=False,
                    system="darwin",
                    machine="arm64",
                    bundle=bundle,
                )
            )
            self.assertFalse(
                can_self_update(
                    info,
                    frozen=True,
                    system="linux",
                    machine="arm64",
                    bundle=bundle,
                )
            )
            self.assertFalse(
                can_self_update(
                    info,
                    frozen=True,
                    system="darwin",
                    machine="x86_64",
                    bundle=bundle,
                )
            )

    def test_update_checks_allowed_respects_env(self) -> None:
        old_skip = os.environ.get("TRACKLAB_SKIP_UPDATE_CHECK")
        old_qt = os.environ.get("QT_QPA_PLATFORM")
        try:
            os.environ["TRACKLAB_SKIP_UPDATE_CHECK"] = "1"
            os.environ.pop("QT_QPA_PLATFORM", None)
            self.assertFalse(update_checks_allowed())
            os.environ.pop("TRACKLAB_SKIP_UPDATE_CHECK", None)
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            self.assertFalse(update_checks_allowed())
        finally:
            if old_skip is None:
                os.environ.pop("TRACKLAB_SKIP_UPDATE_CHECK", None)
            else:
                os.environ["TRACKLAB_SKIP_UPDATE_CHECK"] = old_skip
            if old_qt is None:
                os.environ.pop("QT_QPA_PLATFORM", None)
            else:
                os.environ["QT_QPA_PLATFORM"] = old_qt

    def test_update_json_forces_minimum_and_critical(self) -> None:
        self.assertEqual(
            parse_update_manifest(
                {"version": "0.3.3", "minimum_version": "0.3.0", "critical": False}
            ),
            ("0.3.0", False),
        )
        self.assertEqual(parse_update_manifest({"critical": "false"}), ("", False))
        self.assertEqual(parse_update_manifest("nope"), ("", False))
        self.assertFalse(update_is_required("0.3.1", "0.3.0", False))
        self.assertTrue(update_is_required("0.2.9", "0.3.0", False))
        self.assertTrue(update_is_required("0.3.2", "0.3.0", True))

        forced = evaluate_release(
            _payload(),
            "0.1.0",
            skipped="v0.2.0",
            minimum_version="0.3.0",
        )
        self.assertEqual(forced.status, UpdateStatus.AVAILABLE)
        self.assertIn("必须更新", forced.message)
        self.assertEqual(forced.minimum_version, "0.3.0")

        critical = evaluate_release(
            _payload(),
            "0.1.0",
            skipped="v0.2.0",
            critical=True,
        )
        self.assertEqual(critical.status, UpdateStatus.AVAILABLE)
        self.assertTrue(critical.critical)

        optional = evaluate_release(
            _payload(),
            "0.1.0",
            skipped="v0.2.0",
            minimum_version="0.1.0",
        )
        self.assertEqual(optional.status, UpdateStatus.SKIPPED)

    def test_missing_update_json_keeps_current_behavior(self) -> None:
        manifest = {
            "version": "0.2.0",
            "minimum_version": "0.2.0",
            "critical": True,
        }
        url = (
            "https://github.com/haogesong2011-collab/tracklab/"
            "releases/download/v0.2.0/update.json"
        )

        def with_manifest(url_called, headers, timeout):
            del headers, timeout
            if url_called == GITHUB_API_LATEST:
                payload = _payload()
                payload["assets"] = list(payload["assets"]) + [
                    {
                        "name": "update.json",
                        "browser_download_url": url,
                        "size": 40,
                    }
                ]
                return 200, {}, json.dumps(payload).encode()
            if url_called == url:
                return 200, {}, json.dumps(manifest).encode()
            raise AssertionError(url_called)

        info = check_for_update("0.1.0", skipped="v0.2.0", transport=with_manifest)
        self.assertEqual(info.status, UpdateStatus.AVAILABLE)
        self.assertTrue(info.critical)

        def broken(url_called, headers, timeout):
            del headers, timeout
            if url_called == GITHUB_API_LATEST:
                payload = _payload()
                payload["assets"] = list(payload["assets"]) + [
                    {
                        "name": "update.json",
                        "browser_download_url": url,
                        "size": 40,
                    }
                ]
                return 200, {}, json.dumps(payload).encode()
            return 404, {}, b""

        fallback = check_for_update("0.1.0", skipped="v0.2.0", transport=broken)
        self.assertEqual(fallback.status, UpdateStatus.SKIPPED)
        self.assertFalse(fallback.critical)
        self.assertEqual(fallback.minimum_version, "")

    def test_shortcuts_do_not_describe_shift_click_as_negative(self) -> None:
        text = (ROOT / "app" / "main_window.py").read_text(encoding="utf-8")
        self.assertNotIn("Shift+点击：负点", text)
        self.assertIn("Shift+Control 点击：加点", text)


if __name__ == "__main__":
    unittest.main()
