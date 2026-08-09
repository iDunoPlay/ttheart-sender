from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.exceptions import TemplateNotFoundError
from ttheart_sender.screen.capture import load_image, save_image
from ttheart_sender.screen.matcher import TemplateMatcher
from ttheart_sender.screen.templates import Template, TemplateLibrary


def noisy_background(width=400, height=300, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)


def stamp(canvas, patch, x, y):
    canvas[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
    return canvas


def marker(size=30, color=(30, 200, 240)):
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:, :] = color
    patch[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = (255, 255, 255)
    return patch


def make_template(patch, name="marker"):
    return Template(name=name, path=None, image=patch)  # type: ignore[arg-type]


def test_finds_a_template_at_a_known_position():
    patch = marker()
    scene = stamp(noisy_background(), patch, 120, 80)
    match = TemplateMatcher(confidence=0.9).find(scene, make_template(patch))

    assert match is not None
    assert match.rect.left == 120 and match.rect.top == 80
    assert match.center.as_tuple() == (135, 95)
    assert match.confidence > 0.95


def test_returns_none_when_absent():
    matcher = TemplateMatcher(confidence=0.9)
    assert matcher.find(noisy_background(), make_template(marker())) is None


def test_best_score_reports_similarity_even_on_a_miss():
    matcher = TemplateMatcher(confidence=0.99)
    score = matcher.best_score(noisy_background(), make_template(marker()))
    assert 0.0 <= score < 0.99


def test_find_all_suppresses_overlapping_hits():
    patch = marker()
    scene = noisy_background()
    for x, y in [(20, 20), (200, 40), (100, 200)]:
        stamp(scene, patch, x, y)

    matches = TemplateMatcher(confidence=0.9).find_all(scene, make_template(patch))
    assert len(matches) == 3
    corners = sorted((m.rect.left, m.rect.top) for m in matches)
    assert corners == [(20, 20), (100, 200), (200, 40)]


def test_multi_scale_finds_a_resized_copy():
    import cv2

    patch = marker(40)
    smaller = cv2.resize(patch, (20, 20), interpolation=cv2.INTER_AREA)
    scene = stamp(noisy_background(), smaller, 60, 60)

    matcher = TemplateMatcher(confidence=0.85, scales=[1.0])
    assert matcher.find(scene, make_template(patch)) is None

    multi = TemplateMatcher(confidence=0.85, scales=[0.5, 0.75, 1.0])
    match = multi.find(scene, make_template(patch))
    assert match is not None and match.scale == 0.5


def test_template_larger_than_search_area_is_skipped_not_crashed():
    matcher = TemplateMatcher(confidence=0.5)
    huge = make_template(marker(200))
    assert matcher.find(noisy_background(100, 100), huge) is None


def test_alpha_channel_becomes_a_match_mask(tmp_path):
    import cv2

    # A circular button whose corners are transparent.
    size = 40
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(rgba, (size // 2, size // 2), size // 2 - 2, (40, 180, 255, 255), -1)

    path = tmp_path / "button.png"
    save_image(path, rgba)

    library = TemplateLibrary(tmp_path)
    template = library.get("button")
    assert template.mask is not None

    # Place the circle on a background the transparent corners must not match.
    scene = noisy_background()
    opaque = template.image.copy()
    circle_mask = template.mask > 0
    region = scene[100:140, 150:190]
    region[circle_mask] = opaque[circle_mask]

    match = TemplateMatcher(confidence=0.9, grayscale=False).find(scene, template)
    assert match is not None
    assert match.rect.left == 150 and match.rect.top == 100


def test_template_library_resolution_and_errors(tmp_path):
    save_image(tmp_path / "a.png", marker())
    (tmp_path / "sub").mkdir()
    save_image(tmp_path / "sub" / "b.png", marker())

    library = TemplateLibrary(tmp_path)
    assert sorted(library.available()) == ["a.png", "sub/b.png"]
    assert library.get("a").name == "a.png"
    assert library.get("a.png").name == "a.png"
    assert library.get("sub/b").name == "sub/b.png"
    assert library.get("b").name == "sub/b.png"  # bare-name fallback

    with pytest.raises(TemplateNotFoundError) as exc:
        library.get("missing")
    assert "a.png" in str(exc.value)


def test_image_round_trip_handles_unicode_paths(tmp_path):
    path = tmp_path / "心_button.png"
    original = marker()
    save_image(path, original)
    assert np.array_equal(load_image(path), original)
