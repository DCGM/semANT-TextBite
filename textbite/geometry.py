from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np

from pero_ocr.document_ocr.layout import TextLine


AABB = namedtuple("AABB", "xmin ymin xmax ymax")


def polygon_to_bbox(polygon: np.ndarray) -> AABB:
    mins = np.min(polygon, axis=0)
    maxs = np.max(polygon, axis=0)

    # (minx, miny, maxx, maxy)
    return AABB(mins[0], mins[1], maxs[0], maxs[1])


def bbox_intersection(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    dy = min(lhs.ymax, rhs.ymax) - max(lhs.ymin, rhs.ymin)

    return dx * dy if dx >= 0.0 and dy >= 0.0 else 0.0


def bbox_intersection_over_area(lhs: AABB, rhs: AABB) -> float:
    intersection = bbox_intersection(lhs, rhs)
    area = abs((lhs.xmax - lhs.xmin)) * abs((lhs.ymax - lhs.ymin))

    assert intersection <= area
    return intersection / area


def best_intersecting_bbox(target_bbox: AABB, candidate_bboxes: List[AABB]) -> Optional[int]:
    best_region = None
    best_intersection = 0.0

    for i, bbox in enumerate(candidate_bboxes):
        intersection = bbox_intersection(target_bbox, bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_region = i

    return best_region


def is_contained(lhs: AABB, rhs: AABB, threshold: float=0.9) -> bool:
    return bbox_intersection_over_area(lhs, rhs) >= threshold

# https://stackoverflow.com/a/66801704/9703830
def polygon_area(xs: Tuple[np.int64], ys: Tuple[np.int64]) -> np.float64:
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def polygon_centroid(xs: Tuple[np.int64], ys: Tuple[np.int64]) -> np.ndarray:
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    xy = np.array([xs, ys])
    c = np.dot(xy + np.roll(xy, 1, axis=1),
               xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
               ) / (6 * polygon_area(xs, ys))
    
    return c


def get_lines_polygon(lines: List[TextLine]) -> np.ndarray:
    bboxes = [polygon_to_bbox(line.polygon) for line in lines]
    min_x = min(bboxes, key=lambda x: x.xmin).xmin
    min_y = min(bboxes, key=lambda x: x.ymin).ymin
    max_x = max(bboxes, key=lambda x: x.xmax).xmax
    max_y = max(bboxes, key=lambda x: x.ymax).ymax

    polygon = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ])

    return polygon
