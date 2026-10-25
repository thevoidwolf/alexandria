"""REPORT.md's charts."""
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("matplotlib")

from evals.charts import THEMES, difference_chart, latency_chart, place_labels  # noqa: E402


def test_a_label_with_room_sits_beside_its_point():
    assert place_labels([(100, 100), (100, 300)], gap=12, width=90) == [(110, 100), (110, 300)]


def test_crowded_labels_stack_in_their_points_order_so_leaders_dont_cross():
    # Three points within a label's height of each other, the middle one furthest right.
    out = place_labels([(100, 205), (130, 200), (110, 210)], gap=12, width=90)
    assert {x for x, _ in out} == {150}                     # one column, right of them all
    ys = [y for _, y in out]
    assert ys[1] < ys[0] < ys[2]                            # same order as the points
    assert sorted(ys) == pytest.approx([193, 205, 217])     # gap apart, around their middle


ROWS = [("fts", -0.675, -0.72, -0.63), ("vec", -0.002, -0.006, 0.001), ("big", 0.02, 0.01, 0.03)]
POINTS = [("fts", 0.5, 0.9, 0.047), ("vec", 20.0, 23.4, 0.72), ("hybrid", 21.0, 24.3, 0.722)]


def _render(tmp_path, name):
    difference_chart(ROWS, baseline="hybrid", title="t", subtitle="s", theme=THEMES["dark"],
                     path=tmp_path / f"{name}-ndcg.svg")
    latency_chart(POINTS, baseline="hybrid", title="t", subtitle="s", theme=THEMES["light"],
                  path=tmp_path / f"{name}-latency.svg")
    return [(tmp_path / f"{name}-{kind}.svg").read_bytes() for kind in ("ndcg", "latency")]


def test_charts_are_svg_and_render_the_same_every_time(tmp_path):
    first, again = _render(tmp_path, "a"), _render(tmp_path, "b")
    assert first == again
    for svg in first:
        assert ET.fromstring(svg).tag.endswith("svg")
        assert b"<dc:date>" not in svg
