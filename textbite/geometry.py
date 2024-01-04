from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np

from pero_ocr.document_ocr.layout import TextLine


Point = namedtuple("Point", "x y")


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


def bbox_intersection_x(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    return max(dx, 0.0)


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


def bbox_center(bbox: AABB) -> Point:
    x = (bbox.xmin + ((bbox.xmax - bbox.xmin) / 2))
    y = (bbox.ymin + ((bbox.ymax - bbox.ymin) / 2))
    return Point(x, y)


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


class PageGeometry:
    def __init__(self, bites):
        self.bites = bites
        self.bite_geometries = [BiteGeometry(bite) for bite in self.bites]

        for bite_geometry in self.bite_geometries:
            bite_geometry.set_parent(self.bite_geometries)
            bite_geometry.set_child(self.bite_geometries)


class BiteGeometry:
    def __init__(self, bite):
        self.bite = bite
        self.parent: Optional[BiteGeometry] = None
        self.child: Optional[BiteGeometry] = None

        self.center = bbox_center(self.bite.bbox)

    def children_iterator(self):
        ptr = self.child
        while ptr:
            yield ptr
            ptr = ptr.child

    def parent_iterator(self):
        ptr = self.parent
        while ptr:
            yield ptr
            ptr = ptr.parent

    def set_parent(self, bite_geometries) -> None:
        # Filter regions below me
        parent_candidates = [bg for bg in bite_geometries if bg.center.y < self.center.y]
        # Filter lines that have no horizontal overlap with me
        parent_candidates = [bg for bg in parent_candidates if bbox_intersection_x(self.bite.bbox, bg.bite.bbox)]
        if parent_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the highest Y values
            self.parent = max(parent_candidates, key=lambda x: x.center.y)

    def set_child(self, bite_geometries) -> None:
        # Filter lines above me
        child_candidates = [bg for bg in bite_geometries if bg.center.y > self.center.y]
        # Filter lines that have no horizontal overlap with me
        child_candidates = [bg for bg in child_candidates if bbox_intersection_x(self.bite.bbox, bg.bite.bbox)]
        if child_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the lowest Y values
            self.child = min(child_candidates, key=lambda x: x.center.y)
