#!/usr/bin/env python3
"""
Bridge Digital Twin

Pygame prototype for a human/robot cantilever bridge task.

Model assumptions:
    - The six brick families are continuous prismatic bars.
    - A side-view contour is enough to estimate mass and COM because depth is
      constant for all pieces.
    - Brick and base shapes come from Rhino side-contour vertices. Absolute
      model coordinates are normalized to local coordinates before drawing and
      physics.
    - Static stability is checked at every support interface: the COM of the
      brick plus all bricks above it must fall inside that brick's contact
      interval.

Controls:
    A-F          select brick type
    5 / 9 / 13   set base width preset and reset scene
    Left click   human placement; robot evaluates, may reject, then responds
    Space        robot places the current recommendation
    S            toggle conservative/aggressive strategy
    Z            undo last placement
    R            reset scene
    Q / Esc      quit
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pygame


# Window and world layout ----------------------------------------------------

WIN_W, WIN_H = 1280, 720
PANEL_X = 880
BML, BMR = 52, 16
BBOTTOM = WIN_H - 92

CELL_MM = 7.0
CELL = 1.0
EDGE_TOL = 0.03
GAP_W = 13.0
BASE_PRESETS = (5, 9, 13)

# A positive COM margin close to zero is still physically unsafe: camera
# registration, gripper placement, and material tolerances are all larger than
# a few tenths of a millimetre. One world unit is one 7 mm cell.
MIN_ACCEPT_MARGIN = 0.30
CONSERVATIVE_MARGIN = 0.55


@dataclass
class Layout:
    base_span: int = 9

    @property
    def base_shape(self) -> "BaseType":
        return BASE_TYPES[self.base_span]

    @property
    def base_w(self) -> float:
        return self.base_shape.width

    @property
    def base_label(self) -> int:
        return self.base_span

    @property
    def left_x0(self) -> float:
        return 0.0

    @property
    def right_x0(self) -> float:
        return self.base_w + GAP_W

    @property
    def world_w(self) -> float:
        return self.base_w * 2 + GAP_W

    @property
    def gap_x0(self) -> float:
        return self.base_w

    @property
    def gap_x1(self) -> float:
        return self.base_w + GAP_W

    @property
    def scale(self) -> float:
        return (PANEL_X - BML - BMR) / self.world_w

    @property
    def center_x(self) -> float:
        return self.world_w / 2


LAYOUT = Layout(9)


# Apple-ish restrained palette ---------------------------------------------

BG = (246, 247, 250)
CANVAS_BG = (250, 251, 253)
PANEL_BG = (18, 19, 23)
PANEL_LINE = (48, 50, 58)
TEXT = (29, 31, 36)
MUTED = (112, 118, 130)
PANEL_TEXT = (238, 240, 246)
PANEL_MUTED = (148, 153, 166)
GRID = (224, 227, 234)
BASE = (78, 82, 92)
BASE_TOP = (108, 113, 126)
BLUE = (0, 122, 255)
GREEN = (52, 199, 89)
YELLOW = (255, 204, 0)
ORANGE = (255, 149, 0)
RED = (255, 59, 48)
WHITE = (255, 255, 255)
COM = (255, 45, 85)
CONTACT = (0, 122, 255)

BRICK_COLS: Dict[str, Tuple[int, int, int]] = {
    "A": (219, 107, 70),
    "B": (53, 153, 217),
    "C": (82, 177, 107),
    "D": (198, 78, 132),
    "E": (205, 159, 62),
    "F": (130, 103, 214),
}


# Rhino side-contour coordinates --------------------------------------------

BRICK_COORDS_MM: Dict[str, List[Tuple[float, float]]] = {
    "A": [
        (-102.5, 14.25833), (-99, 18.588457), (-92, 18.588457),
        (-88.5, 14.25833), (-92, 9.928203), (-88.5, 5.598076),
        (-92, 1.267949), (-99, 1.267949), (-102.5, 5.598076),
        (-109.5, 5.598076), (-113, 1.267949), (-120, 1.267949),
        (-123.5, 5.598076), (-130.5, 5.598076), (-134, 9.928203),
        (-130.5, 14.25833), (-123.5, 14.25833), (-120, 9.928203),
        (-113, 9.928203), (-109.5, 14.25833),
    ],
    "B": [
        (-131, -15.071797), (-124, -15.071797), (-120.5, -10.74167),
        (-113.5, -10.74167), (-110, -15.071797), (-103, -15.071797),
        (-99.5, -10.74167), (-92.5, -10.74167), (-89, -15.071797),
        (-82, -15.071797), (-78.5, -19.401924), (-82, -23.732051),
        (-89, -23.732051), (-92.5, -19.401924), (-99.5, -19.401924),
        (-103, -23.732051), (-110, -23.732051), (-113.5, -19.401924),
        (-120.5, -19.401924), (-124, -23.732051), (-131, -23.732051),
        (-134.5, -19.401924),
    ],
    "C": [
        (-133, -53.732051), (-126, -53.732051), (-122.5, -49.401924),
        (-115.5, -49.401924), (-112, -45.071797), (-105, -45.071797),
        (-101.5, -49.401924), (-94.5, -49.401924), (-91, -53.732051),
        (-84.0, -53.732051), (-80.5, -49.401924), (-84.0, -45.071797),
        (-91, -45.071797), (-94.5, -40.74167), (-101.5, -40.74167),
        (-105, -36.411543), (-112, -36.411543), (-115.5, -40.74167),
        (-122.5, -40.74167), (-126, -45.071797), (-133, -45.071797),
        (-136.5, -49.401924),
    ],
    "D": [
        (-55.5, 5.598076), (-59, 1.267949), (-66, 1.267949),
        (-69.5, 5.598076), (-66, 9.928203), (-59, 9.928203),
        (-55.5, 14.25833), (-48.5, 14.25833), (-45, 9.928203),
        (-38, 9.928203), (-34.5, 14.25833), (-27.5, 14.25833),
        (-24, 9.928203), (-17, 9.928203), (-13.5, 14.25833),
        (-6.5, 14.25833), (-3, 9.928203), (-6.5, 5.598076),
        (-13.5, 5.598076), (-17, 1.267949), (-24, 1.267949),
        (-27.5, 5.598076), (-34.5, 5.598076), (-38, 1.267949),
        (-45, 1.267949), (-48.5, 5.598076),
    ],
    "E": [
        (-13, -23.732051), (-6, -23.732051), (-2.5, -19.401924),
        (-6, -15.071797), (-13, -15.071797), (-16.5, -10.74167),
        (-23.5, -10.74167), (-27, -6.411543), (-34, -6.411543),
        (-37.5, -10.74167), (-44.5, -10.74167), (-48, -15.071797),
        (-55, -15.071797), (-58.5, -10.74167), (-65.5, -10.74167),
        (-69, -15.071797), (-65.5, -19.401924), (-58.5, -19.401924),
        (-55, -23.732051), (-48, -23.732051), (-44.5, -19.401924),
        (-37.5, -19.401924), (-34, -15.071797), (-27, -15.071797),
        (-23.5, -19.401924), (-16.5, -19.401924),
    ],
    "F": [
        (-70, -48.732051), (-66.5, -53.062178), (-59.5, -53.062178),
        (-56, -48.732051), (-49, -48.732051), (-45.5, -44.401924),
        (-38.5, -44.401924), (-35, -48.732051), (-28, -48.732051),
        (-24.5, -44.401924), (-17.5, -44.401924), (-14, -48.732051),
        (-7, -48.732051), (-3.5, -53.062178), (3.5, -53.062178),
        (7.0, -48.732051), (3.5, -44.401924), (-3.5, -44.401924),
        (-7, -40.071797), (-14, -40.071797), (-17.5, -35.74167),
        (-24.5, -35.74167), (-28, -40.071797), (-35, -40.071797),
        (-38.5, -35.74167), (-45.5, -35.74167), (-49, -40.071797),
        (-56, -40.071797), (-59.5, -44.401924), (-66.5, -44.401924),
    ],
}

BASE_COORDS_MM: Dict[int, List[Tuple[float, float]]] = {
    5: [
        (19.041443, 13.420933), (68.041443, 13.420933),
        (68.041443, 24.370506), (61.041443, 24.370506),
        (57.541443, 28.700634), (50.541443, 28.700634),
        (47.041443, 24.370506), (40.041443, 24.370506),
        (36.541443, 28.700634), (29.541443, 28.700634),
        (26.041443, 24.370506), (19.041443, 24.370506),
    ],
    9: [
        (20.041443, -37.579067), (111.041443, -37.579067),
        (111.041443, -26.629494), (104.041443, -26.629494),
        (100.536499, -22.293209), (93.541443, -22.299366),
        (90.041443, -26.629494), (83.041443, -26.629494),
        (79.541443, -22.299366), (72.541443, -22.299366),
        (69.041443, -26.629494), (62.041443, -26.629494),
        (58.541443, -22.299366), (51.541443, -22.299366),
        (48.041443, -26.629494), (41.041443, -26.629494),
        (37.541443, -22.299366), (30.541443, -22.299366),
        (27.041443, -26.629494), (20.041443, -26.629494),
    ],
    13: [
        (20.041443, -52.629494), (27.041443, -52.629494),
        (30.541443, -48.299366), (37.541443, -48.299366),
        (41.041443, -52.629494), (48.041443, -52.629494),
        (51.541443, -48.299366), (58.541443, -48.299366),
        (62.041443, -52.629494), (69.041443, -52.629494),
        (72.541443, -48.299366), (79.541443, -48.299366),
        (83.041443, -52.629494), (90.041443, -52.629494),
        (93.541443, -48.299366), (100.541443, -48.299366),
        (104.041443, -52.629494), (111.041443, -52.629494),
        (114.541443, -48.299366), (121.536499, -48.293209),
        (125.041443, -52.629494), (132.041443, -52.629494),
        (135.541443, -48.299366), (142.536499, -48.293209),
        (146.041443, -52.629494), (153.041443, -52.629494),
        (153.041443, -63.579067), (20.041443, -63.579067),
    ],
}

BRICK_SEQ = "ABCDEF"


def normalize_poly(coords_mm: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    min_x = min(x for x, _y in coords_mm)
    min_y = min(y for _x, y in coords_mm)
    pts = tuple(((x - min_x) / CELL_MM, (y - min_y) / CELL_MM) for x, y in coords_mm)
    if pts[0] != pts[-1]:
        pts = pts + (pts[0],)
    return pts


def centroid_area(poly: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    area2 = cx6 = cz6 = 0.0
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        cross = x0 * z1 - x1 * z0
        area2 += cross
        cx6 += (x0 + x1) * cross
        cz6 += (z0 + z1) * cross
    area = abs(area2) / 2.0
    if area < 1e-9:
        return 0.0, 0.0, 0.0
    return abs(cx6) / (6.0 * area), abs(cz6) / (6.0 * area), area


@dataclass(frozen=True)
class BrickType:
    key: str
    poly: Tuple[Tuple[float, float], ...]
    top_edges: Tuple[Tuple[float, float, float], ...]
    bottom_edges: Tuple[Tuple[float, float, float], ...]
    width: float
    height: float
    cx: float
    cz: float
    mass: float


@dataclass(frozen=True)
class BaseType:
    span: int
    poly: Tuple[Tuple[float, float], ...]
    top_edges: Tuple[Tuple[float, float, float], ...]
    width: float
    height: float


def horizontal_edges(
    poly: Sequence[Tuple[float, float]],
    centroid_z: float,
    want: str,
) -> Tuple[Tuple[float, float, float], ...]:
    edges: List[Tuple[float, float, float]] = []
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        if abs(z0 - z1) > EDGE_TOL or abs(x0 - x1) < EDGE_TOL:
            continue
        mid_z = (z0 + z1) / 2
        if want == "top" and mid_z < centroid_z:
            continue
        if want == "bottom" and mid_z > centroid_z:
            continue
        edges.append((min(x0, x1), max(x0, x1), (z0 + z1) / 2))
    return tuple(sorted(edges))


def make_brick(key: str, coords_mm: Sequence[Tuple[float, float]]) -> BrickType:
    poly = normalize_poly(coords_mm)
    cx, cz, area = centroid_area(poly)
    width = max(x for x, _z in poly)
    height = max(z for _x, z in poly)
    top_edges = horizontal_edges(poly, cz, "top")
    bottom_edges = horizontal_edges(poly, cz, "bottom")
    return BrickType(key, poly, top_edges, bottom_edges, width, height, cx, cz, area)


def make_base(span: int, coords_mm: Sequence[Tuple[float, float]]) -> BaseType:
    poly = normalize_poly(coords_mm)
    cx, cz, _area = centroid_area(poly)
    width = max(x for x, _z in poly)
    height = max(z for _x, z in poly)
    return BaseType(span, poly, horizontal_edges(poly, cz, "top"), width, height)


BTYPES: Dict[str, BrickType] = {k: make_brick(k, v) for k, v in BRICK_COORDS_MM.items()}
BASE_TYPES: Dict[int, BaseType] = {k: make_base(k, v) for k, v in BASE_COORDS_MM.items()}


# Structure model ------------------------------------------------------------

BASE_LEFT = -1
BASE_RIGHT = -2


@dataclass
class SurfaceSeg:
    x0: float
    x1: float
    z: float
    owner: int


@dataclass
class BrickInst:
    id: int
    btype: BrickType
    wx: float
    wz: float
    actor: str
    contacts: List[SurfaceSeg] = field(default_factory=list)
    margin: float = math.inf

    @property
    def x0(self) -> float:
        return self.wx

    @property
    def x1(self) -> float:
        return self.wx + self.btype.width

    @property
    def world_cx(self) -> float:
        return self.wx + self.btype.cx

    @property
    def world_cz(self) -> float:
        return self.wz + self.btype.cz

    @property
    def contact_x0(self) -> float:
        return min((c.x0 for c in self.contacts), default=self.x0)

    @property
    def contact_x1(self) -> float:
        return max((c.x1 for c in self.contacts), default=self.x1)


@dataclass
class JointResult:
    brick: BrickInst
    mass: float
    comx: float
    sx0: float
    sx1: float
    margin: float
    above_ids: Tuple[int, ...]


def base_surfaces() -> List[SurfaceSeg]:
    base = LAYOUT.base_shape
    surfaces: List[SurfaceSeg] = []
    for x0, owner in ((LAYOUT.left_x0, BASE_LEFT), (LAYOUT.right_x0, BASE_RIGHT)):
        for ex0, ex1, ez in base.top_edges:
            surfaces.append(SurfaceSeg(x0 + ex0, x0 + ex1, ez, owner))
    return surfaces


def top_surfaces(brick: BrickInst) -> List[SurfaceSeg]:
    return [
        SurfaceSeg(brick.wx + x0, brick.wx + x1, brick.wz + z, brick.id)
        for x0, x1, z in brick.btype.top_edges
    ]


def overlap(a0: float, a1: float, b0: float, b1: float) -> Optional[Tuple[float, float]]:
    x0, x1 = max(a0, b0), min(a1, b1)
    if x1 - x0 > 1e-6:
        return x0, x1
    return None


def world_poly(bt: BrickType, wx: float, wz: float) -> Tuple[Tuple[float, float], ...]:
    return tuple((wx + x, wz + z) for x, z in bt.poly)


def _orient(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_segment_cross(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
    eps: float = 1e-7,
) -> bool:
    oa0 = _orient(a0, a1, b0)
    oa1 = _orient(a0, a1, b1)
    ob0 = _orient(b0, b1, a0)
    ob1 = _orient(b0, b1, a1)
    return oa0 * oa1 < -eps and ob0 * ob1 < -eps


def _point_strictly_inside_poly(pt: Tuple[float, float], poly: Sequence[Tuple[float, float]]) -> bool:
    x, z = pt
    inside = False
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        if abs(_orient((x0, z0), (x1, z1), pt)) < 1e-7:
            if min(x0, x1) - 1e-7 <= x <= max(x0, x1) + 1e-7 and min(z0, z1) - 1e-7 <= z <= max(z0, z1) + 1e-7:
                return False
        if (z0 > z) != (z1 > z):
            x_at_z = (x1 - x0) * (z - z0) / (z1 - z0) + x0
            if x_at_z > x + 1e-7:
                inside = not inside
    return inside


def _signed_area(poly: Sequence[Tuple[float, float]]) -> float:
    return sum(
        poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]
        for i in range(len(poly) - 1)
    ) / 2.0


def _point_in_triangle(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    eps: float = 1e-7,
) -> bool:
    o1 = _orient(a, b, p)
    o2 = _orient(b, c, p)
    o3 = _orient(c, a, p)
    return o1 >= -eps and o2 >= -eps and o3 >= -eps


def triangulate_poly(poly: Sequence[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    pts = list(poly[:-1] if poly[0] == poly[-1] else poly)
    if _signed_area(pts + [pts[0]]) < 0:
        pts.reverse()

    triangles: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
    guard = 0
    while len(pts) > 3 and guard < 500:
        guard += 1
        ear_found = False
        for i in range(len(pts)):
            prev_pt = pts[(i - 1) % len(pts)]
            pt = pts[i]
            next_pt = pts[(i + 1) % len(pts)]
            if _orient(prev_pt, pt, next_pt) <= 1e-7:
                continue
            if any(
                _point_in_triangle(other, prev_pt, pt, next_pt)
                for j, other in enumerate(pts)
                if j not in {(i - 1) % len(pts), i, (i + 1) % len(pts)}
            ):
                continue
            triangles.append((prev_pt, pt, next_pt))
            del pts[i]
            ear_found = True
            break
        if not ear_found:
            break
    if len(pts) == 3:
        triangles.append((pts[0], pts[1], pts[2]))
    return triangles


def convex_polys_overlap_area(
    a: Sequence[Tuple[float, float]],
    b: Sequence[Tuple[float, float]],
    eps: float = 1e-7,
) -> bool:
    for poly in (a, b):
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            axis = (-(p1[1] - p0[1]), p1[0] - p0[0])
            amin = min(p[0] * axis[0] + p[1] * axis[1] for p in a)
            amax = max(p[0] * axis[0] + p[1] * axis[1] for p in a)
            bmin = min(p[0] * axis[0] + p[1] * axis[1] for p in b)
            bmax = max(p[0] * axis[0] + p[1] * axis[1] for p in b)
            if min(amax, bmax) - max(amin, bmin) <= eps:
                return False
    return True


def _vertical_fill_intervals(
    poly: Sequence[Tuple[float, float]],
    x: float,
    eps: float = 1e-7,
) -> List[Tuple[float, float]]:
    zs: List[float] = []
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        if abs(x0 - x1) <= eps:
            continue
        if min(x0, x1) + eps < x < max(x0, x1) - eps:
            t = (x - x0) / (x1 - x0)
            zs.append(z0 + t * (z1 - z0))
    zs.sort()
    return [(zs[i], zs[i + 1]) for i in range(0, len(zs) - 1, 2)]


def _vertical_overlap_len(
    a_intervals: Sequence[Tuple[float, float]],
    b_intervals: Sequence[Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    total = 0.0
    for a0, a1 in a_intervals:
        for b0, b1 in b_intervals:
            total += max(0.0, min(a1, b1) - max(a0, b0) - eps)
    return total


def sampled_overlap_area(
    a: Sequence[Tuple[float, float]],
    b: Sequence[Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    x0 = max(min(p[0] for p in a), min(p[0] for p in b))
    x1 = min(max(p[0] for p in a), max(p[0] for p in b))
    if x1 - x0 <= eps:
        return 0.0

    # Rhino contours are built on a 7 mm pitch. 0.025 world units is 0.175 mm,
    # fine enough to reject visible overlap while keeping recommendation search
    # responsive.
    step = 0.025
    samples = max(1, int(math.ceil((x1 - x0) / step)))
    dx = (x1 - x0) / samples
    area = 0.0
    for i in range(samples):
        x = x0 + (i + 0.5) * dx
        overlap_len = _vertical_overlap_len(
            _vertical_fill_intervals(a, x, eps),
            _vertical_fill_intervals(b, x, eps),
            eps,
        )
        area += overlap_len * dx
    return area


def _owner_world_poly(owner: int, bricks_by_id: Dict[int, BrickInst]) -> Optional[Tuple[Tuple[float, float], ...]]:
    if owner == BASE_LEFT:
        return tuple((LAYOUT.left_x0 + x, z) for x, z in LAYOUT.base_shape.poly)
    if owner == BASE_RIGHT:
        return tuple((LAYOUT.right_x0 + x, z) for x, z in LAYOUT.base_shape.poly)
    brick = bricks_by_id.get(owner)
    if brick:
        return world_poly(brick.btype, brick.wx, brick.wz)
    return None


def _boundary_contact_segments(
    upper: Sequence[Tuple[float, float]],
    lower: Sequence[Tuple[float, float]],
    owner: int,
    z_fallback: float,
    eps: float = 0.04,
) -> List[SurfaceSeg]:
    x0 = max(min(p[0] for p in upper), min(p[0] for p in lower))
    x1 = min(max(p[0] for p in upper), max(p[0] for p in lower))
    if x1 - x0 <= eps:
        return []

    step = 0.025
    samples = max(1, int(math.ceil((x1 - x0) / step)))
    dx = (x1 - x0) / samples
    runs: List[Tuple[float, float]] = []
    run_start: Optional[float] = None
    last_x: Optional[float] = None

    for i in range(samples):
        x = x0 + (i + 0.5) * dx
        upper_intervals = _vertical_fill_intervals(upper, x)
        lower_intervals = _vertical_fill_intervals(lower, x)
        touching = False
        for ub, _ut in upper_intervals:
            for _lb, lt in lower_intervals:
                if ub >= lt - eps and abs(ub - lt) <= eps:
                    touching = True
                    break
            if touching:
                break

        if touching:
            if run_start is None:
                run_start = x - dx / 2
            last_x = x + dx / 2
        elif run_start is not None and last_x is not None:
            runs.append((run_start, last_x))
            run_start = None
            last_x = None

    if run_start is not None and last_x is not None:
        runs.append((run_start, last_x))

    return [
        SurfaceSeg(max(x0, a), min(x1, b), z_fallback, owner)
        for a, b in runs
        if b - a > 0.05
    ]


def expanded_contacts(
    bt: BrickType,
    wx: float,
    wz: float,
    contacts: Sequence[SurfaceSeg],
    existing: Sequence[BrickInst],
) -> List[SurfaceSeg]:
    if not contacts:
        return []
    candidate_poly = world_poly(bt, wx, wz)
    bricks_by_id = {b.id: b for b in existing}
    expanded: List[SurfaceSeg] = []
    for owner in sorted({c.owner for c in contacts}):
        owner_poly = _owner_world_poly(owner, bricks_by_id)
        owner_contacts = [c for c in contacts if c.owner == owner]
        fallback_z = max(c.z for c in owner_contacts)
        if owner_poly is None:
            expanded.extend(owner_contacts)
            continue
        boundary = _boundary_contact_segments(candidate_poly, owner_poly, owner, fallback_z)
        expanded.extend(boundary or owner_contacts)
    return merge_contacts(expanded)


def polys_overlap_area(
    a: Sequence[Tuple[float, float]],
    b: Sequence[Tuple[float, float]],
) -> bool:
    eps = 1e-7
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    az0, az1 = min(p[1] for p in a), max(p[1] for p in a)
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    bz0, bz1 = min(p[1] for p in b), max(p[1] for p in b)
    if ax1 <= bx0 + eps or bx1 <= ax0 + eps or az1 <= bz0 + eps or bz1 <= az0 + eps:
        return False

    return sampled_overlap_area(a, b, eps) > 1e-4


def local_bottom_at(bt: BrickType, lx: float) -> Optional[float]:
    for x0, x1, bz in bt.bottom_edges:
        if x0 - 1e-9 <= lx <= x1 + 1e-9:
            return bz
    return None


def compute_drop(bt: BrickType, wx: float, surfaces: Sequence[SurfaceSeg]) -> Tuple[Optional[float], List[SurfaceSeg]]:
    best = -math.inf
    contacts: List[SurfaceSeg] = []

    for sx in surfaces:
        for bx0, bx1, bbz in bt.bottom_edges:
            hit = overlap(wx + bx0, wx + bx1, sx.x0, sx.x1)
            if not hit:
                continue
            cand = sx.z - bbz
            cseg = SurfaceSeg(hit[0], hit[1], sx.z, sx.owner)
            if cand > best + 1e-9:
                best = cand
                contacts = [cseg]
            elif abs(cand - best) <= 1e-9:
                contacts.append(cseg)

    if best == -math.inf:
        return None, []
    return best, merge_contacts(contacts)


def merge_contacts(contacts: Sequence[SurfaceSeg]) -> List[SurfaceSeg]:
    if not contacts:
        return []
    contacts = sorted(contacts, key=lambda c: (c.owner, c.z, c.x0))
    merged: List[SurfaceSeg] = []
    cur = contacts[0]
    for c in contacts[1:]:
        if c.owner == cur.owner and abs(c.z - cur.z) < 1e-9 and c.x0 <= cur.x1 + 1e-6:
            cur = SurfaceSeg(cur.x0, max(cur.x1, c.x1), cur.z, cur.owner)
        else:
            merged.append(cur)
            cur = c
    merged.append(cur)
    return merged


@dataclass
class Rec:
    side: str
    bt: BrickType
    wx: float
    wz: float
    contacts: List[SurfaceSeg]
    score: float
    margin_after: float
    dm: float
    reach_gain: float
    intent: str


@dataclass
class Bridge:
    bricks: List[BrickInst] = field(default_factory=list)
    joints: List[JointResult] = field(default_factory=list)
    next_id: int = 1

    def clone(self) -> "Bridge":
        copied = Bridge(next_id=self.next_id)
        copied.bricks = [
            BrickInst(b.id, b.btype, b.wx, b.wz, b.actor, list(b.contacts), b.margin)
            for b in self.bricks
        ]
        copied.analyse()
        return copied

    def all_surfaces(self) -> List[SurfaceSeg]:
        surfs = base_surfaces()
        for b in self.bricks:
            surfs.extend(top_surfaces(b))
        return surfs

    def try_place(self, bt: BrickType, wx: float, actor: str) -> Optional[BrickInst]:
        wz, contacts = compute_drop(bt, wx, self.all_surfaces())
        if wz is None or contact_width(contacts) < 0.2:
            return None
        candidate_poly = world_poly(bt, wx, wz)
        for existing in self.bricks:
            if polys_overlap_area(candidate_poly, world_poly(existing.btype, existing.wx, existing.wz)):
                return None
        contacts = expanded_contacts(bt, wx, wz, contacts, self.bricks)
        if contact_width(contacts) < 0.2:
            return None
        return BrickInst(self.next_id, bt, wx, wz, actor, contacts)

    def place(self, bt: BrickType, wx: float, actor: str) -> Optional[BrickInst]:
        brick = self.try_place(bt, wx, actor)
        if not brick:
            return None
        self.bricks.append(brick)
        self.next_id += 1
        self.analyse()
        return brick

    def remove_id(self, bid: int) -> None:
        self.bricks = [b for b in self.bricks if b.id != bid]
        self.analyse()

    def undo_last(self) -> Optional[BrickInst]:
        if not self.bricks:
            return None
        b = self.bricks.pop()
        self.analyse()
        return b

    def reset(self) -> None:
        self.bricks.clear()
        self.joints.clear()
        self.next_id = 1

    def analyse(self) -> None:
        by_id = {b.id: b for b in self.bricks}
        children: Dict[int, List[int]] = {b.id: [] for b in self.bricks}
        for b in self.bricks:
            for owner in {c.owner for c in b.contacts if c.owner > 0}:
                if owner in children:
                    children[owner].append(b.id)

        def support_fractions(b: BrickInst) -> Dict[int, float]:
            widths: Dict[int, float] = {}
            for c in b.contacts:
                widths[c.owner] = widths.get(c.owner, 0.0) + max(0.0, c.x1 - c.x0)
            total = sum(widths.values())
            if total <= 1e-9:
                return {}
            return {owner: width / total for owner, width in widths.items()}

        cache: Dict[int, Tuple[float, float, Tuple[int, ...]]] = {}

        def carried_load(bid: int) -> Tuple[float, float, Tuple[int, ...]]:
            """
            Resultant load carried by this brick before it is transferred to
            its own supports. Multi-support children split their load across
            supports by contact width, avoiding duplicate full-mass counting.
            """
            if bid in cache:
                return cache[bid]
            b = by_id[bid]
            mass = b.btype.mass
            moment = b.world_cx * mass
            ids = [bid]
            for cid in children.get(bid, []):
                child = by_id[cid]
                fractions = support_fractions(child)
                frac = fractions.get(bid, 0.0)
                if frac <= 0.0:
                    continue
                cmass, cmoment, cids = carried_load(cid)
                mass += cmass * frac
                moment += cmoment * frac
                ids.extend(cids)
            result = (mass, moment, tuple(sorted(set(ids))))
            cache[bid] = result
            return result

        results: List[JointResult] = []
        for b in self.bricks:
            mass, moment, ids = carried_load(b.id)
            comx = moment / mass
            sx0, sx1 = b.contact_x0, b.contact_x1
            margin = min(comx - sx0, sx1 - comx)
            b.margin = margin
            results.append(JointResult(b, mass, comx, sx0, sx1, margin, ids))
        self.joints = results

    @property
    def min_margin(self) -> float:
        return min((j.margin for j in self.joints), default=math.inf)

    @property
    def danger(self) -> float:
        if not self.joints:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.min_margin / 1.25))

    @property
    def left_reach(self) -> float:
        return max(
            (
                b.x1 - LAYOUT.gap_x0
                for b in self.bricks
                if (b.x0 + b.x1) / 2 < LAYOUT.center_x and b.x1 > LAYOUT.gap_x0
            ),
            default=0.0,
        )

    @property
    def right_reach(self) -> float:
        return max(
            (
                LAYOUT.gap_x1 - b.x0
                for b in self.bricks
                if (b.x0 + b.x1) / 2 >= LAYOUT.center_x and b.x0 < LAYOUT.gap_x1
            ),
            default=0.0,
        )

    @property
    def gap_coverage(self) -> float:
        return min(1.0, (self.left_reach + self.right_reach) / GAP_W)

    @property
    def bridge_closed(self) -> bool:
        left_front = max(
            (b.x1 for b in self.bricks if (b.x0 + b.x1) / 2 < LAYOUT.center_x),
            default=-math.inf,
        )
        right_front = min(
            (b.x0 for b in self.bricks if (b.x0 + b.x1) / 2 >= LAYOUT.center_x),
            default=math.inf,
        )
        return left_front >= right_front - 0.15


def contact_width(contacts: Sequence[SurfaceSeg]) -> float:
    return sum(max(0.0, c.x1 - c.x0) for c in contacts)


def side_for_x(wx: float) -> str:
    return "L" if wx < LAYOUT.center_x else "R"


def reach_for_side(brick: BrickInst, side: str) -> float:
    if side == "L":
        return max(0.0, brick.x1 - LAYOUT.gap_x0)
    return max(0.0, LAYOUT.gap_x1 - brick.x0)


def snap_place_x(wx: float) -> float:
    """All legal placements are on integer 7 mm cell steps."""
    return float(math.floor(wx + 0.5))


def recommend(bridge: Bridge, strategy: str, top_n: int = 4) -> List[Rec]:
    base_margin = bridge.min_margin if bridge.joints else 1.0
    required_margin = CONSERVATIVE_MARGIN if strategy == "conservative" else MIN_ACCEPT_MARGIN
    cands: List[Rec] = []
    x_min = -math.ceil(max(b.width for b in BTYPES.values()))
    x_max = math.ceil(LAYOUT.world_w)

    for side in ("L", "R"):
        for bt in BTYPES.values():
            for xi in range(x_min, x_max + 1):
                wx = float(xi)
                center = wx + bt.width / 2
                if side_for_x(center) != side:
                    continue
                sim = bridge.clone()
                bi = sim.place(bt, wx, "robot")
                if bi:
                    margin_after = sim.min_margin
                    dm = margin_after - base_margin
                    reach = reach_for_side(bi, side)
                    width = contact_width(bi.contacts)
                    if margin_after >= required_margin:
                        intent = "reinforce" if dm > 0.08 and reach < 0.75 else "cantilever"
                        if strategy == "conservative":
                            score = 1.25 * margin_after + 0.45 * width + 0.35 * reach
                        else:
                            score = 0.45 * margin_after + 0.30 * width + 1.20 * reach
                        if intent == "reinforce" and strategy == "conservative":
                            score += 0.25
                        cands.append(Rec(side, bt, wx, bi.wz, bi.contacts, score, margin_after, dm, reach, intent))

    cands.sort(key=lambda r: (-r.score, r.side, r.bt.key, r.wx))
    return cands[:top_n]


def evaluate_human(before: Bridge, after: Bridge, placed: BrickInst) -> Tuple[str, str]:
    if after.min_margin < 0.0:
        return "remove", "Robot rejects: COM outside support"
    if after.min_margin < MIN_ACCEPT_MARGIN:
        return "remove", "Robot rejects: margin below tolerance"
    if not before.joints:
        return "nod", "Robot nods: stable opening move"
    if after.min_margin + 1e-6 >= before.min_margin:
        return "nod", "Robot nods: stability improved"
    if after.min_margin > MIN_ACCEPT_MARGIN:
        return "nod", "Robot nods: risky but acceptable"
    return "shake", "Robot shakes: weak support margin"


# Drawing helpers ------------------------------------------------------------

def w2p(wx: float, wz: float) -> Tuple[int, int]:
    return int(BML + wx * LAYOUT.scale), int(BBOTTOM - wz * LAYOUT.scale)


def p2w(px: int, py: int) -> Tuple[float, float]:
    return (px - BML) / LAYOUT.scale, (BBOTTOM - py) / LAYOUT.scale


def brick_points(bt: BrickType, wx: float, wz: float) -> List[Tuple[int, int]]:
    return [w2p(wx + x, wz + z) for x, z in bt.poly]


def lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def margin_color(m: float) -> Tuple[int, int, int]:
    if m == math.inf:
        return BLUE
    if m >= 0.85:
        return GREEN
    if m >= 0.25:
        return lerp(YELLOW, GREEN, (m - 0.25) / 0.6)
    if m >= 0.0:
        return lerp(RED, YELLOW, m / 0.25)
    return RED


def poly(sc: pygame.Surface, pts: Sequence[Tuple[int, int]], fill: Tuple[int, int, int, int], outline: Tuple[int, int, int], width: int = 2) -> None:
    if len(pts) < 3:
        return
    layer = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.polygon(layer, fill, pts)
    sc.blit(layer, (0, 0))
    pygame.draw.polygon(sc, outline, pts, width)


class UI:
    def __init__(self, screen: pygame.Surface):
        self.sc = screen
        self.f12 = pygame.font.Font(None, 18)
        self.f14 = pygame.font.Font(None, 21)
        self.f16 = pygame.font.Font(None, 24)
        self.f20 = pygame.font.Font(None, 30)
        self.f28 = pygame.font.Font(None, 42)

    def render(
        self,
        bridge: Bridge,
        recs: List[Rec],
        sel: str,
        strategy: str,
        message: str,
        reaction: str,
        hover_wx: Optional[float],
    ) -> None:
        self.sc.fill(BG)
        self.draw_canvas(bridge, recs, sel, hover_wx, message, reaction)
        self.draw_panel(bridge, recs, sel, strategy, message, reaction)
        pygame.display.flip()

    def draw_canvas(self, bridge: Bridge, recs: List[Rec], sel: str, hover_wx: Optional[float], message: str, reaction: str) -> None:
        pygame.draw.rect(self.sc, CANVAS_BG, (0, 0, PANEL_X, WIN_H))

        for z in range(0, 17):
            y = w2p(0, z)[1]
            if 20 < y < BBOTTOM:
                pygame.draw.line(self.sc, GRID, (BML, y), (PANEL_X - BMR, y), 1)

        gx0, _ = w2p(LAYOUT.gap_x0, 0)
        gx1, _ = w2p(LAYOUT.gap_x1, 0)
        gap_layer = pygame.Surface((gx1 - gx0, WIN_H), pygame.SRCALPHA)
        gap_layer.fill((0, 122, 255, 16))
        self.sc.blit(gap_layer, (gx0, 0))

        self.draw_base(LAYOUT.left_x0, BASE_LEFT)
        self.draw_base(LAYOUT.right_x0, BASE_RIGHT)

        if recs:
            r = recs[0]
            self.draw_ghost(r.bt, r.wx, r.wz, 82)

        if hover_wx is not None and sel in BTYPES:
            bt = BTYPES[sel]
            wx = snap_place_x(hover_wx - bt.width / 2)
            preview = bridge.try_place(bt, wx, "human")
            if preview:
                self.draw_ghost(bt, wx, preview.wz, 42)

        for b in bridge.bricks:
            self.draw_brick(b)

        for b in bridge.bricks:
            for c in b.contacts:
                y = w2p(0, c.z)[1]
                pygame.draw.line(self.sc, CONTACT, (w2p(c.x0, 0)[0], y), (w2p(c.x1, 0)[0], y), 3)

        if bridge.joints:
            self.draw_weakest(min(bridge.joints, key=lambda j: j.margin))

        pygame.draw.line(self.sc, (196, 201, 211), (BML, BBOTTOM), (PANEL_X - BMR, BBOTTOM), 2)
        self.draw_dimension_labels()

        if message:
            col = {"nod": GREEN, "shake": ORANGE, "remove": RED}.get(reaction, BLUE)
            self.banner(message, col)
        if bridge.bridge_closed:
            self.banner("Bridge closed", GREEN, y_offset=-56)

    def draw_base(self, x0: float, owner: int) -> None:
        base = LAYOUT.base_shape
        pts = [w2p(x0 + x, z) for x, z in base.poly]
        poly(self.sc, pts, BASE + (228,), tuple(max(0, c - 34) for c in BASE), 2)
        for ex0, ex1, ez in base.top_edges:
            y = w2p(0, ez)[1]
            pygame.draw.line(self.sc, BASE_TOP, (w2p(x0 + ex0, 0)[0], y), (w2p(x0 + ex1, 0)[0], y), 3)
        label = self.f12.render(f"{base.span}-span", True, WHITE)
        px = w2p(x0 + base.width / 2, 0)[0]
        py = w2p(0, base.height * 0.35)[1]
        self.sc.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

    def draw_brick(self, b: BrickInst) -> None:
        pts = brick_points(b.btype, b.wx, b.wz)
        base = BRICK_COLS[b.btype.key]
        fill = lerp(base, margin_color(b.margin), 0.28)
        outline = tuple(max(0, c - 58) for c in base)
        alpha = 224 if b.actor == "human" else 204
        poly(self.sc, pts, fill + (alpha,), outline)
        cx, cz = w2p(b.world_cx, b.world_cz)
        pygame.draw.circle(self.sc, WHITE, (cx, cz), 5)
        pygame.draw.circle(self.sc, COM, (cx, cz), 3)
        label = self.f12.render(b.btype.key.lower() if b.actor == "human" else b.btype.key, True, (18, 20, 24))
        self.sc.blit(label, (cx - label.get_width() // 2, cz - 18))

    def draw_ghost(self, bt: BrickType, wx: float, wz: float, alpha: int) -> None:
        pts = brick_points(bt, wx, wz)
        col = BRICK_COLS[bt.key]
        poly(self.sc, pts, col + (alpha,), col, 2)

    def draw_weakest(self, joint: JointResult) -> None:
        b = joint.brick
        y = w2p(0, max(c.z for c in b.contacts))[1]
        x0, x1 = w2p(joint.sx0, 0)[0], w2p(joint.sx1, 0)[0]
        pulse = int(96 + 70 * math.sin(pygame.time.get_ticks() / 260))
        layer = pygame.Surface((max(1, x1 - x0), 7), pygame.SRCALPHA)
        layer.fill((*margin_color(joint.margin), pulse))
        self.sc.blit(layer, (x0, y - 4))
        cx = w2p(joint.comx, 0)[0]
        pygame.draw.line(self.sc, COM, (cx, y - 28), (cx, y + 8), 2)
        label = self.f12.render("load COM", True, COM)
        self.sc.blit(label, (cx - label.get_width() // 2, y - 44))

    def draw_dimension_labels(self) -> None:
        items = [
            (LAYOUT.left_x0, LAYOUT.base_w, BASE_TOP, f"base {LAYOUT.base_label}"),
            (LAYOUT.gap_x0, GAP_W, BLUE, "gap"),
            (LAYOUT.right_x0, LAYOUT.base_w, BASE_TOP, f"base {LAYOUT.base_label}"),
        ]
        for x0, width, col, name in items:
            label = self.f12.render(f"{name}", True, col)
            px = w2p(x0 + width / 2, 0)[0]
            self.sc.blit(label, (px - label.get_width() // 2, BBOTTOM + 16))

    def banner(self, text: str, col: Tuple[int, int, int], y_offset: int = 0) -> None:
        label = self.f20.render(text, True, WHITE)
        pad_x, pad_y = 18, 9
        w, h = label.get_width() + pad_x * 2, label.get_height() + pad_y * 2
        x = (PANEL_X - w) // 2
        y = 24 + y_offset
        layer = pygame.Surface((w, h), pygame.SRCALPHA)
        layer.fill((*col, 220))
        self.sc.blit(layer, (x, y))
        self.sc.blit(label, (x + pad_x, y + pad_y))

    def draw_panel(self, bridge: Bridge, recs: List[Rec], sel: str, strategy: str, message: str, reaction: str) -> None:
        pygame.draw.rect(self.sc, PANEL_BG, (PANEL_X, 0, WIN_W - PANEL_X, WIN_H))
        pygame.draw.line(self.sc, PANEL_LINE, (PANEL_X, 0), (PANEL_X, WIN_H), 1)
        x = PANEL_X + 22
        y = 24
        width = WIN_W - PANEL_X - 44

        def row(text: str, font=None, col=PANEL_TEXT, gap=6) -> None:
            nonlocal y
            surf = (font or self.f14).render(text, True, col)
            self.sc.blit(surf, (x, y))
            y += surf.get_height() + gap

        def sep(space=12) -> None:
            nonlocal y
            y += 4
            pygame.draw.line(self.sc, PANEL_LINE, (x, y), (x + width, y), 1)
            y += space

        def bar(frac: float, col: Tuple[int, int, int]) -> None:
            nonlocal y
            frac = max(0.0, min(1.0, frac))
            pygame.draw.rect(self.sc, (44, 46, 54), (x, y, width, 10), border_radius=5)
            pygame.draw.rect(self.sc, col, (x, y, int(width * frac), 10), border_radius=5)
            y += 20

        row("Bridge Twin", self.f28, PANEL_TEXT, 4)
        row(f"side contour model / {CELL_MM:.0f} mm pitch", self.f12, PANEL_MUTED)
        sep()

        strat_col = ORANGE if strategy == "aggressive" else GREEN
        row(f"Strategy: {strategy}", self.f16, strat_col)
        row(f"Base preset: {LAYOUT.base_label}-span   keys 5/9/3", self.f12, PANEL_MUTED)
        sep()

        danger_col = lerp(GREEN, RED, bridge.danger)
        mm = "--" if bridge.min_margin == math.inf else f"{bridge.min_margin:.2f}"
        row("Stability", self.f16, PANEL_TEXT)
        bar(bridge.danger, danger_col)
        row(f"danger {bridge.danger * 100:.0f}%   min margin {mm}", self.f12, danger_col)
        row(f"coverage {bridge.gap_coverage * 100:.0f}%   L {bridge.left_reach:.1f} / R {bridge.right_reach:.1f}", self.f12, PANEL_MUTED)
        sep()

        row("Robot reaction", self.f16, PANEL_TEXT)
        if message:
            col = {"nod": GREEN, "shake": ORANGE, "remove": RED}.get(reaction, BLUE)
            row(message, self.f12, col)
        else:
            row("Waiting for human placement", self.f12, PANEL_MUTED)
        sep()

        row("Next recommendation", self.f16, PANEL_TEXT)
        if not recs:
            row("No stable candidate", self.f12, RED)
        for i, r in enumerate(recs[:3]):
            col = BLUE if i == 0 else PANEL_TEXT
            row(f"{i + 1}. {r.side}  brick {r.bt.key}  x={r.wx:.1f}  {r.intent}", self.f12, col, 3)
            row(f"   margin {r.margin_after:.2f}   reach +{r.reach_gain:.1f}", self.f12, PANEL_MUTED, 5)
        sep()

        row("Brick library", self.f16, PANEL_TEXT)
        bx = x
        by = y
        for k in BRICK_SEQ:
            bt = BTYPES[k]
            bw = int(34 + bt.width * 8)
            rect = pygame.Rect(bx, by, bw, 30)
            pygame.draw.rect(self.sc, BRICK_COLS[k], rect, border_radius=7)
            if k == sel:
                pygame.draw.rect(self.sc, WHITE, rect, 2, border_radius=7)
            label = self.f14.render(k, True, WHITE)
            self.sc.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))
            bx += bw + 8
            if bx > WIN_W - 70:
                bx = x
                by += 38
        y = by + 48
        sep()

        for line in (
            "Left click: human move",
            "Space: robot places ghost",
            "S strategy   Z undo   R reset",
            "A-F brick   Q quit",
        ):
            row(line, self.f12, PANEL_MUTED, 4)


def robot_place_recommendation(bridge: Bridge, strategy: str) -> Tuple[List[Rec], str, str]:
    recs = recommend(bridge, strategy)
    if not recs:
        return recs, "Robot waits: no stable move", "shake"
    r = recs[0]
    placed = bridge.place(r.bt, r.wx, "robot")
    if not placed:
        return recommend(bridge, strategy), "Robot move failed", "shake"
    return recommend(bridge, strategy), f"Robot placed {r.bt.key} on {r.side}", "nod"


def set_base_width(span: int, bridge: Bridge) -> None:
    LAYOUT.base_span = span
    bridge.reset()


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Bridge Digital Twin")
    clock = pygame.time.Clock()

    bridge = Bridge()
    ui = UI(screen)
    selected = "A"
    strategy = "conservative"
    message = ""
    reaction = ""
    recs = recommend(bridge, strategy)

    while True:
        mx, my = pygame.mouse.get_pos()
        hover_wx = p2w(mx, my)[0] if mx < PANEL_X else None

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.KEYDOWN:
                key = ev.unicode.upper()
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if key in BTYPES:
                    selected = key
                elif key in {"5", "9"}:
                    set_base_width(int(key), bridge)
                    message = f"Base reset to {key}-span"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                elif ev.key == pygame.K_3:
                    set_base_width(13, bridge)
                    message = "Base reset to 13-span"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                elif ev.key == pygame.K_s:
                    strategy = "aggressive" if strategy == "conservative" else "conservative"
                    recs = recommend(bridge, strategy)
                    message = f"Strategy changed to {strategy}"
                    reaction = ""
                elif ev.key == pygame.K_SPACE:
                    recs, message, reaction = robot_place_recommendation(bridge, strategy)
                elif ev.key == pygame.K_z:
                    removed = bridge.undo_last()
                    message = f"Undo {removed.btype.key}" if removed else "Nothing to undo"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                elif ev.key == pygame.K_r:
                    bridge.reset()
                    message = "Scene reset"
                    reaction = ""
                    recs = recommend(bridge, strategy)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and mx < PANEL_X:
                bt = BTYPES[selected]
                wx = snap_place_x(p2w(mx, my)[0] - bt.width / 2)
                before = bridge.clone()
                placed = bridge.place(bt, wx, "human")
                if not placed:
                    message = "No support below that placement"
                    reaction = "shake"
                else:
                    reaction, message = evaluate_human(before, bridge, placed)
                    if reaction == "remove":
                        bridge.remove_id(placed.id)
                        recs = recommend(bridge, strategy)
                    else:
                        recs, robot_message, robot_reaction = robot_place_recommendation(bridge, strategy)
                        message = f"{message}; {robot_message}"
                        reaction = robot_reaction if robot_reaction == "shake" else reaction

        ui.render(bridge, recs, selected, strategy, message, reaction, hover_wx)
        clock.tick(30)


if __name__ == "__main__":
    main()
