from __future__ import annotations

from typing import Optional, List, Tuple, Dict
from collections import namedtuple
from math import sqrt
from functools import cached_property

import numpy as np
import torch

from shapely.ops import nearest_points
from shapely.geometry import Polygon

from pero_ocr.document_ocr.layout import PageLayout, TextLine


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


def bbox_to_yolo(bbox: AABB, page_width, page_height) -> Tuple[float, float, float, float]:
    dx, dy = bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin
    x = (bbox.xmin + (dx / 2.0)) / page_width
    y = (bbox.ymin + (dy / 2.0)) / page_height
    width = dx / page_width
    height = dy / page_height

    return x, y, width, height


class Ray:
    def __init__(self, origin: Point, direction: Point):
        self.origin = origin
        self.direction = direction

        length = sqrt(self.direction.x*self.direction.x + self.direction.y*self.direction.y)
        x = self.direction.x / length
        y = self.direction.y / length
        self.direction = Point(x, y)

    def intersects_bbox(self, bbox: AABB) -> Optional[float]:
        if self.direction.x == 0 and (self.origin.x < bbox.xmin or self.origin.x > bbox.xmax):
            return None

        if self.direction.y == 0 and (self.origin.y < bbox.ymin or self.origin.y > bbox.ymax):
            return None

        tmin = -float('inf')
        tmax = float('inf')

        if self.direction.x != 0:
            tx1 = (bbox.xmin - self.origin.x) / self.direction.x
            tx2 = (bbox.xmax - self.origin.x) / self.direction.x

            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if self.direction.y != 0:
            ty1 = (bbox.ymin - self.origin.y) / self.direction.y
            ty2 = (bbox.ymax - self.origin.y) / self.direction.y

            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmin <= tmax and tmax >= 0:
            return max(tmin, 0)
        else:
            return None


def find_visible_entities(rays: List[Ray], entities: List[GeometryEntity]) -> List[GeometryEntity]:
    visible_entities = []

    for ray in rays:
        best_dist = float("inf")
        closest_entity = None
        for entity in entities:
            dist = ray.intersects_bbox(entity.bbox)
            if dist:
                if dist < best_dist:
                    best_dist = dist
                    closest_entity = entity

        if closest_entity and closest_entity not in visible_entities:
            visible_entities.append(closest_entity)

    return visible_entities


class GeometryEntity:
    def __init__(self, page_geometry: Optional[PageGeometry]=None):
        self.page_geometry = page_geometry # Reference to the geometry of the entire page

        self.parent: Optional[GeometryEntity] = None
        self.child: Optional[GeometryEntity] = None
        self.neighbourhood: Optional[List[GeometryEntity]] = None
        self.visible_entities: Optional[List[GeometryEntity]] = None

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

    def set_visibility(self, entities: List[GeometryEntity]) -> None:
        ...


class RegionGeometry(GeometryEntity):
    def __init__(self, bbox: AABB, page_geometry: Optional[PageGeometry]):
        super().__init__(page_geometry)
        self._bbox = bbox

    @property
    def bbox(self) -> AABB:
        assert self._bbox.xmax > self._bbox.xmin and self._bbox.ymax > self._bbox.ymin
        return self._bbox
    
    def set_visibility(self, entities: List[GeometryEntity]) -> None:
        assert self.page_geometry is not None

        self.visible_entities = []
        other_entities = [entity for entity in entities if self is not entity]

        if self.parent is not None:
            self.visible_entities.append(self.parent)

        if self.child is not None:
            self.visible_entities.append(self.child)

        # Create horizontal rays
        horizontal_rays = []
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, 0.5)))
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, 0)))
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, -0.5)))

        horizontal_visible_entities = find_visible_entities(horizontal_rays, other_entities)
        self.visible_entities.extend(horizontal_visible_entities)
        for ve in horizontal_visible_entities:
            for relative in ve.lineage_iterator():
                if relative not in self.visible_entities:
                    self.visible_entities.append(relative)

        self.visible_entities = list(set(self.visible_entities))


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

    @property
    def page_area(self):
        return self.page_width * self.page_height

    @property
    def avg_line_width(self) -> float:
        if not self.lines or len(self.lines) == 0:
            raise ValueError("No lines exist in this PageGeometry.")
        return sum(line.get_width() for line in self.lines) / len(self.lines)

    @property
    def avg_line_height(self) -> float:
        if not self.lines or len(self.lines) == 0:
            raise ValueError("No lines exist in this PageGeometry.")
        return sum(line.get_height() for line in self.lines) / len(self.lines)
    
    @property
    def line_heads(self) -> List[LineGeometry]:
        return [line_geometry for line_geometry in self.lines if line_geometry.parent is None]
    
    @property
    def avg_line_distance_y(self) -> float:
        processed_pairs = []
        distance_sum = 0.0

        for line_geometry in self.lines:
            if line_geometry.parent is not None:
                if set([line_geometry, line_geometry.parent]) not in processed_pairs:
                    distance_sum += bbox_dist_y(line_geometry.bbox, line_geometry.parent.bbox)
                    processed_pairs.append(set([line_geometry, line_geometry.parent]))

            if line_geometry.child is not None:
                if set([line_geometry, line_geometry.child]) not in processed_pairs:
                    distance_sum += bbox_dist_y(line_geometry.bbox, line_geometry.child.bbox)
                    processed_pairs.append(set([line_geometry, line_geometry.child]))

        return distance_sum / len(processed_pairs)
