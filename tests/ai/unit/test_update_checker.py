"""Update checker logic. Tests never call the live GitHub API."""

from __future__ import annotations

import json
import os
import sys
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
    check_for_update,
    evaluate_release,
    is_newer,
    parse_version,
    safe_release_url,
    should_auto_check,
    update_checks_allowed,
)


def _payload(**overrides):
    data = {
        "tag_name": "v0.2.0",
        "draft": False,
        "prerelease": False,
        "html_url": "https://github.com/haogesong2011-collab/tracklab/releases/tag/v0.2.0",
        "body": "修复更新检查",
        "published_at": "2026-09-01T12:00:00Z",
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


if __name__ == "__main__":
    unittest.main()
