from __future__ import annotations

from typing import Optional, List
from collections import namedtuple
from math import sqrt
from functools import cached_property

import numpy as np

from pero_ocr.core.layout import PageLayout, TextLine


Point = namedtuple("Point", "x y")
AABB = namedtuple("AABB", "xmin ymin xmax ymax")


def dist_l2(p1: Point, p2: Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return sqrt(dx*dx + dy*dy)


def bbox_dist_y(bbox1: AABB, bbox2: AABB) -> float:
    bbox1_center_y = bbox_center(bbox1).y
    bbox2_center_y = bbox_center(bbox2).y

    bbox1_half_height = bbox1.ymax - bbox1_center_y
    bbox2_half_height = bbox2.ymax - bbox2_center_y

    return max(0.0, abs(bbox1_center_y - bbox2_center_y) - bbox1_half_height - bbox2_half_height)


def polygon_to_bbox(polygon: np.ndarray) -> AABB:
    mins = np.min(polygon, axis=0)
    maxs = np.max(polygon, axis=0)

    # (minx, miny, maxx, maxy)
    return AABB(int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))


def enclosing_bbox(bboxes: List[AABB]) -> AABB:
    xmins = [bbox.xmin for bbox in bboxes]
    xmaxs = [bbox.xmax for bbox in bboxes]
    ymins = [bbox.ymin for bbox in bboxes]
    ymaxs = [bbox.ymax for bbox in bboxes]

    bbox = AABB(min(xmins), max(xmaxs), min(ymins), max(ymaxs))
    return bbox


def bbox_center(bbox: AABB) -> Point:
    x = (bbox.xmin + ((bbox.xmax - bbox.xmin) / 2))
    y = (bbox.ymin + ((bbox.ymax - bbox.ymin) / 2))

    return Point(x, y)


def bbox_area(bbox: AABB) -> float:
    return float((bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin))


def bbox_intersection(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    dy = min(lhs.ymax, rhs.ymax) - max(lhs.ymin, rhs.ymin)

    return dx * dy if dx >= 0.0 and dy >= 0.0 else 0.0


def bbox_intersection_over_area(lhs: AABB, rhs: AABB) -> float:
    intersection = bbox_intersection(lhs, rhs)
    area = bbox_area(lhs)

    assert intersection <= area
    return intersection / area


def bbox_intersection_x(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    return max(dx, 0.0)


def best_intersecting_bbox(target_bbox: AABB, candidate_bboxes: List[AABB]):
    best_region = None
    best_intersection = 0.0
    for i, bbox in enumerate(candidate_bboxes):
        intersection = bbox_intersection(target_bbox, bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_region = i

    return best_region


class GeometryEntity:
    def __init__(self, page_geometry: Optional[PageGeometry]=None):
        self.page_geometry = page_geometry # Reference to the geometry of the entire page

        self.parent: Optional[GeometryEntity] = None
        self.child: Optional[GeometryEntity] = None

    @property
    def bbox(self) -> AABB:
        ...
        
    @property
    def width(self) -> float:
        return self.bbox.xmax - self.bbox.xmin

    @property
    def height(self) -> float:
        return self.bbox.ymax - self.bbox.ymin

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)

    @property
    def bbox_area(self):
        return bbox_area(self.bbox)
    
    @property
    def number_of_predecessors(self) -> int:
        return sum([1 for _ in self.parent_iterator()])

    @property
    def number_of_successors(self) -> int:
        return sum([1 for _ in self.children_iterator()])
    
    @property
    def vertical_neighbours(self, neighbourhood_size: int) -> List[LineGeometry]:
        neighbourhood = [self]
        parent_ptr = self.parent
        child_ptr = self.child

        for _ in range(neighbourhood_size):
            if parent_ptr:
                neighbourhood.append(parent_ptr)
                parent_ptr = parent_ptr.parent

            if child_ptr:
                neighbourhood.append(child_ptr)
                child_ptr = child_ptr.child

        return neighbourhood

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

    def lineage_iterator(self):
        for parent in self.parent_iterator():
            yield parent
        for child in self.children_iterator():
            yield child            
    
    def set_parent(self, entities: List[GeometryEntity], threshold: float=0.0) -> None:
        parent_candidates = [entity for entity in entities if self is not entity]
        # Filter entities below me
        parent_candidates = [entity for entity in parent_candidates if entity.center.y < self.center.y]
        # Filter entities that have no horizontal overlap with me
        parent_candidates = [entity for entity in parent_candidates if bbox_intersection_x(self.bbox, entity.bbox) > threshold]
        if parent_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the highest Y values
            self.parent = max(parent_candidates, key=lambda x: x.center.y)

    def set_child(self, entities: List[GeometryEntity], threshold: int=0.0) -> None:
        child_candidates = [entity for entity in entities if self is not entity]
        # Filter entities above me
        child_candidates = [entity for entity in child_candidates if entity.center.y > self.center.y]
        # Filter entities that have no horizontal overlap with me
        child_candidates = [entity for entity in child_candidates if bbox_intersection_x(self.bbox, entity.bbox) > threshold]
        if child_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the lowest Y values
            self.child = min(child_candidates, key=lambda x: x.center.y)


class RegionGeometry(GeometryEntity):
    def __init__(self, bbox: AABB, page_geometry: Optional[PageGeometry]):
        super().__init__(page_geometry)
        self._bbox = bbox

    @property
    def bbox(self) -> AABB:
        assert self._bbox.xmax > self._bbox.xmin and self._bbox.ymax > self._bbox.ymin
        return self._bbox


class LineGeometry(GeometryEntity):
    def __init__(self, text_line: TextLine, page_geometry: Optional[PageGeometry]):
        super().__init__(page_geometry)

        self.text_line: TextLine = text_line
        self.polygon = text_line.polygon

    @cached_property
    def bbox(self) -> AABB:
        _bbox = polygon_to_bbox(self.text_line.polygon)
        assert _bbox.xmax > _bbox.xmin and _bbox.ymax > _bbox.ymin
        return _bbox
    

class PageGeometry:
    def __init__(
            self,
            regions: List[AABB]=[],
            path: Optional[str]=None,
            pagexml: Optional[PageLayout]=None,
        ):
        self.pagexml: PageLayout = pagexml
        if path:
            self.pagexml = PageLayout(file=path)

        self.lines = []
        self.regions = [RegionGeometry(region, self) for region in regions]

        if self.pagexml is not None:
            self.lines: List[LineGeometry] = [LineGeometry(line, self) for line in self.pagexml.lines_iterator() if line.transcription and line.transcription.strip()]
            self.lines_by_id = {line.text_line.id: line for line in self.lines}

            h, w = self.pagexml.page_size
            self.page_width = w
            self.page_height = h
        
        for line in self.lines:
            line.set_parent(self.lines)
            line.set_child(self.lines)

        for region in self.regions:
            region.set_parent(self.regions, threshold=10)
            region.set_child(self.regions, threshold=10)

