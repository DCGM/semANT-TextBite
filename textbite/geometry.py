from __future__ import annotations

from typing import Optional, List, Tuple, Dict
from collections import namedtuple
from math import sqrt

import numpy as np
import torch

from shapely.ops import nearest_points
from shapely.geometry import Polygon

from pero_ocr.document_ocr.layout import PageLayout, TextLine

from textbite.utils import hash_strings


Point = namedtuple("Point", "x y")
AABB = namedtuple("AABB", "xmin ymin xmax ymax")


def dist_l2(p1: Point, p2: Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return sqrt(dx*dx + dy*dy)


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


def bbox_intersection_x(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    return max(dx, 0.0)


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

    def intersects_bbox(self, bbox: AABB):
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


def find_visible_lines(rays: List[Ray], lines: List[LineGeometry]) -> List[LineGeometry]:
    visible_lines = []

    for ray in rays:
        best_dist = float("inf")
        closest_line = None
        for line in lines:
            dist = ray.intersects_bbox(line.bbox)
            if dist:
                if dist < best_dist:
                    best_dist = dist
                    closest_line = line

        if closest_line and closest_line not in visible_lines:
            visible_lines.append(closest_line)

    return visible_lines


class LineGeometry:
    def __init__(self, text_line: TextLine, page_geometry: Optional[PageGeometry]=None):
        self.text_line: TextLine = text_line
        self.page_geometry: PageGeometry = page_geometry # Reference to the geometry of the entire page
        self.polygon = text_line.polygon

        self.parent: Optional[LineGeometry] = None
        self.child: Optional[LineGeometry] = None
        self.neighbourhood: Optional[List[LineGeometry]] = None
        self.visible_lines: Optional[List[LineGeometry]] = None

        self.bbox: AABB = self.get_bbox()
        assert self.bbox.xmax > self.bbox.xmin and self.bbox.ymax > self.bbox.ymin

        self.center: Point = self.get_center()

    @property
    def bbox_area(self):
        return (self.bbox.xmax - self.bbox.xmin) * (self.bbox.ymax - self.bbox.ymin)

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

    def get_number_of_predecessors(self) -> int:
        return sum([1 for _ in self.parent_iterator()])

    def get_number_of_successors(self) -> int:
        return sum([1 for _ in self.children_iterator()])

    def get_vertical_neighbours(self, neighbourhood_size: int) -> List[LineGeometry]:
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

    def get_bbox(self) -> AABB:
        return polygon_to_bbox(self.text_line.polygon)

    def get_width(self) -> float:
        return self.bbox.xmax - self.bbox.xmin

    def get_height(self) -> float:
        return self.bbox.ymax - self.bbox.ymin

    def get_center(self) -> Point:
        x = (self.bbox.xmin + ((self.bbox.xmax - self.bbox.xmin) / 2))
        y = (self.bbox.ymin + ((self.bbox.ymax - self.bbox.ymin) / 2))
        return Point(x, y)

    def set_parent(self, lines: List[LineGeometry]) -> None:
        # Filter lines below me
        parent_candidates = [line for line in lines if line.center.y < self.center.y]
        # Filter lines that have no horizontal overlap with me
        parent_candidates = [line for line in parent_candidates if bbox_intersection_x(self.bbox, line.bbox)]
        if parent_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the highest Y values
            self.parent = max(parent_candidates, key=lambda x: x.center.y)

    def set_child(self, lines: List[LineGeometry]) -> None:
        # Filter lines above me
        child_candidates = [line for line in lines if line.center.y > self.center.y]
        # Filter lines that have no horizontal overlap with me
        child_candidates = [line for line in child_candidates if bbox_intersection_x(self.bbox, line.bbox)]
        if child_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the lowest Y values
            self.child = min(child_candidates, key=lambda x: x.center.y)

    def set_neighbourhood(self, lines: List[LineGeometry], max_neighbours: int, cached_distances: Dict[str, float]={}) -> None:
        distances = []
        for other_line in lines:
            if self is other_line:
                continue
            hash = hash_strings(self.text_line.id, other_line.text_line.id)
            try:
                distance = cached_distances[hash]
            except KeyError:
                nps = nearest_points(Polygon(self.text_line.polygon), Polygon(other_line.text_line.polygon))
                p1 = Point(nps[0].x, nps[0].y)
                p2 = Point(nps[1].x, nps[1].y)
                distance = dist_l2(p1, p2)
                cached_distances[hash] = distance

            distances.append((other_line, distance))

        self.neighbourhood = []
        mx = max_neighbours if len(distances) > max_neighbours else len(distances)
        while len(self.neighbourhood) != mx:
            m = min(distances, key=lambda x: x[1])
            self.neighbourhood.append(m[0])
            distances.remove(m)

    def set_visible_lines(self, lines: List[LineGeometry]) -> None:
        self.visible_lines = []
        N_HORIZONTAL_RAYS = 100
        y_span = self.page_geometry.page_height * 0.1
        y_step = y_span / N_HORIZONTAL_RAYS
        other_lines = [l for l in lines if not self is l]

        # Create vertical rays
        vertical_rays = []
        mid_point_x = self.bbox.xmin + ((self.bbox.xmax - self.bbox.xmin) / 2.0)
        epsilon = (self.bbox.xmax - self.bbox.xmin) * 0.05
        vertical_rays.append(Ray(Point(self.bbox.xmin + epsilon, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(mid_point_x, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(self.bbox.xmax - epsilon, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(self.bbox.xmin + epsilon, self.center.y), Point(0, -1)))
        vertical_rays.append(Ray(Point(mid_point_x, self.center.y), Point(0, -1)))
        vertical_rays.append(Ray(Point(self.bbox.xmax - epsilon, self.center.y), Point(0, -1)))

        # Add all lines intersected by vertical rays to visible_lines
        vertical_visible_lines = find_visible_lines(vertical_rays, other_lines)
        self.visible_lines.extend(vertical_visible_lines)

        # Create horizontal rays
        horizontal_rays = []
        mid_point_y = self.bbox.ymin + ((self.bbox.ymax - self.bbox.ymin) / 2.0)

        # HORIZONTAL RAYS (legacy)
        # horizontal_rays.append(Ray(Point(bbox.xmax, mid_point_y), Point(1, 1)))
        # horizontal_rays.append(Ray(Point(bbox.xmax, mid_point_y), Point(1, 0)))
        # horizontal_rays.append(Ray(Point(bbox.xmax, mid_point_y), Point(1, -1)))
        # horizontal_visible_lines = find_visible_lines(horizontal_rays, other_lines)
        # horizontal_visible_lines = [l for l in horizontal_visible_lines if l not in self.visible_lines]
        # self.visible_lines.extend(horizontal_visible_lines)
        # for l in horizontal_visible_lines:
        #     for ll in l.parent_iterator():
        #         if ll not in self.visible_lines:
        #             self.visible_lines.append(ll)
        #     for ll in l.children_iterator():
        #         if ll not in self.visible_lines:
        #             self.visible_lines.append(ll)

        # Horizontal rays (current)
        y_start = 0.0 - (y_span / 2.0)
        y = y_start
        for _ in range(N_HORIZONTAL_RAYS):
            horizontal_rays.append(Ray(Point(self.bbox.xmax, mid_point_y), Point(70, y)))
            y += y_step

        horizontal_visible_lines = find_visible_lines(horizontal_rays, other_lines)
        self.visible_lines.extend(horizontal_visible_lines)
        for l in horizontal_visible_lines:
            if l.child and l.child not in self.visible_lines and self is not l.child:
                self.visible_lines.append(l.child)
            if l.parent and l.parent not in self.visible_lines and self is not l.parent:
                self.visible_lines.append(l.parent)

        self.visible_lines = list(set(self.visible_lines))

    def get_features(self, return_type: Optional[str]=None) -> List | np.ndarray | torch.FloatTensor:
        assert self.page_geometry, "Cannot determine features without data of the entire page"
        PLACEHOLDER_VALUE = -100.0

        features = []

        width = self.get_width()
        height = self.get_height()

        #  1. Line width relative to it's neighbourhood average
        vertical_neighbourhood = self.get_vertical_neighbours(2)
        vertical_neighbourhood_avg_width = sum([l.get_width() for l in vertical_neighbourhood]) / len(vertical_neighbourhood)
        features.append(width / vertical_neighbourhood_avg_width)

        #  2. Line width relative to page average
        #  3. Line height relative to page average
        features.append(width / self.page_geometry.get_avg_line_width())
        features.append(height / self.page_geometry.get_avg_line_height())

        # 10. Line width relative to page width
        # 11. Line height relative to page height
        features.append(width / self.page_geometry.page_width)
        features.append(height / self.page_geometry.page_height)

        # 12. Center X coordinate relative to page width
        # 13. Center Y coordinate relative to page height
        features.append(self.center.x / self.page_geometry.page_width)
        features.append(self.center.y / self.page_geometry.page_height)

        # 22. Left X coordinate relative to page width
        # 23. Right X coordinate relative to page width
        # 24. Top Y coordinate relative to page height
        # 25. Bottom Y coordinate relative to page height
        features.append(self.bbox.xmin / self.page_geometry.page_width)
        features.append(self.bbox.xmax / self.page_geometry.page_width)
        features.append(self.bbox.ymin / self.page_geometry.page_height)
        features.append(self.bbox.ymax / self.page_geometry.page_height)

        # 21. Line width relative to line height
        features.append(width / height)

        # 18. Bounding box area relative to page area
        features.append(self.bbox_area / self.page_geometry.page_area)

        #  4. L2 distance from parent (center - center)
        #  6. L2 distance from parent (center - center) relative to my height
        # 14. Distance from parent in Y axis (center - center)
        # 16. Distance from parent in X axis (center - center)
        # 19. Bounding box area relative to parent bounding box area
        if self.parent:
            distance_to_parent = dist_l2(self.center, self.parent.center)
            features.append(distance_to_parent)
            features.append(distance_to_parent / height)
            features.append(abs(self.center.y - self.parent.center.y))
            features.append(abs(self.center.x - self.parent.center.x))
            features.append(self.bbox_area / self.parent.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  5. L2 distance from child (center - center)
        #  7. L2 distance from child (center - center) relative to my height
        # 15. Distance from child in Y axis (center - center)
        # 17. Distance from child in X axis (center - center)
        # 20. Bounding box area relative to child bounding box area
        if self.child:
            distance_to_child = dist_l2(self.center, self.child.center)
            features.append(distance_to_child)
            features.append(distance_to_child / height)
            features.append(abs(self.center.y - self.child.center.y))
            features.append(abs(self.center.x - self.child.center.x))
            features.append(self.bbox_area / self.child.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  8. Number of predecessors
        features.append(self.get_number_of_predecessors())

        #  9. Number of children
        features.append(self.get_number_of_successors())

        match return_type:
            case "pt":
                return torch.tensor(features, dtype=torch.float32)

            case "np":
                return np.array(features, dtype=np.float32)

            case _:
                return features

    def __str__(self) -> str:
        output_str = "LineGeometry object"
        output_str += f"\nCenter = {self.center}"
        output_str += f"\nWidth, height = {self.get_width()}, {self.get_height()}"
        output_str += f"\nBBox = {self.bbox}"
        output_str += f"\nNumber of parents = {self.get_number_of_predecessors()}"
        output_str += f"\nNumber of children = {self.get_number_of_successors()}"

        return output_str


class PageGeometry:
    def __init__(self, path: Optional[str]=None, pagexml: Optional[PageLayout]=None):
        self.pagexml: PageLayout = pagexml
        if path:
            self.pagexml = PageLayout(file=path)

        self.lines: List[LineGeometry] = [LineGeometry(line, self) for line in self.pagexml.lines_iterator() if line.transcription and line.transcription.strip()]
        for line in self.lines:
            line.set_parent(self.lines)
            line.set_child(self.lines)

        self.lines_by_id = {line.text_line.id: line for line in self.lines}

        h, w = self.pagexml.page_size
        self.page_width = w
        self.page_height = h

    @property
    def page_area(self):
        return self.page_width * self.page_height

    def get_avg_line_width(self) -> float:
        return sum(line.get_width() for line in self.lines) / len(self.lines)

    def get_avg_line_height(self) -> float:
        return sum(line.get_height() for line in self.lines) / len(self.lines)

    def set_neighbourhoods(self, max_neighbours: int=10) -> None:
        cached_distances = {}
        for line in self.lines:
            other_lines = [l for l in self.lines if not (l is line)]
            line.set_neighbourhood(other_lines, max_neighbours, cached_distances)

    def set_visibility(self) -> None:
        for line in self.lines:
            other_lines = [l for l in self.lines if not (l is line)]
            line.set_visible_lines(other_lines)

    def __str__(self) -> str:
        output_str = f"PageGeometry object"
        output_str += f"\n{len(self.lines)} lines."
        output_str += f"\nPage size is {self.page_width}x{self.page_height}"

        return output_str


if __name__ == "__main__":
    EXAMPLE_PATH = r"/home/martin/textbite/tmp/cesky-zapad-(2)-08.xml"
    pagexml = PageLayout(file=EXAMPLE_PATH)

    geometry1 = PageGeometry(pagexml=pagexml)

    geometry1.set_neighbourhoods()
    geometry1.set_visibility()

    for line in geometry1.lines:
        print(len(line.visible_lines))
