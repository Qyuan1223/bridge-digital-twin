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
    T            flip selected brick 180 degrees (except D)
    5 / 9 / 13   set base width preset and reset scene
    Left click   human placement; robot evaluates, may reject, then responds
    Space        robot places the current recommendation
    S            toggle conservative/aggressive strategy
    Z            undo last placement
    R            reset scene
    Q / Esc      quit
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
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
X_GRID = 0.5
# Hex-lattice moves project to half-cell offsets in x when the brick steps
# diagonally in z. This is not a half-edge contact: candidate validation still
# requires complete lattice edge coincidence.
PLACE_GRID = 0.5
HEX_RISE_MM = 4.330127018922193
HEX_RISE = HEX_RISE_MM / CELL_MM
EDGE_TOL = 0.03
ALIGN_TOL = 0.04
CONTACT_TOL = 0.08
# Area below this is treated as contour/scan noise, not true material overlap.
# In world units, 0.09 is still visibly smaller than a real half-cell intrusion,
# but it keeps Rhino-derived edge coincidences from being rejected as overlap.
OVERLAP_AREA_TOL = 0.09
GAP_MM = 140.005
GAP_W = GAP_MM / CELL_MM
OUTER_MARGIN_W = 8.0
BASE_PRESETS = (5, 9, 13)

# A positive COM margin close to zero is still physically unsafe: camera
# registration, gripper placement, and material tolerances are all larger than
# a few tenths of a millimetre. One world unit is one 7 mm cell.
MIN_ACCEPT_MARGIN = 0.30
CONSERVATIVE_MARGIN = 0.55
AGGRESSIVE_PREP_MARGIN = 0.85
CONSERVATIVE_PREP_MARGIN = 1.15
AGGRESSIVE_FOUNDATION_TARGET = 0.68
CONSERVATIVE_FOUNDATION_TARGET = 0.70
LOW_LAYER_CLEARANCE = 1.15
AGGRESSIVE_LAYER_REAR_BRICKS = 1
CLOSING_MAX_GAP_W = 5.0
CONSERVATIVE_CLOSING_MAX_GAP_W = 3.0
CONSERVATIVE_CLOSE_PRIORITY_GAP_W = 6.0
MAX_BRICKS = 50
CACHE_LIMIT = 512
PLACE_RULE_VERSION = 5
SINGLE_LAYER_MAX_BRICK_WIDTH = 9.75
MIN_SUPPORT_CONTACT_WIDTH = 1.6
MIN_SUPPORT_CONTACT_RATIO = 0.24
MIN_BASE_CONTACT_RATIO = 0.38

PLACE_OPTIONS_CACHE: Dict[Tuple, List[Tuple[float, List["SurfaceSeg"]]]] = {}
RECOMMEND_CACHE: Dict[Tuple, List["Rec"]] = {}
LOG_PATH = Path(__file__).with_name("bridge_log.txt")


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
        return OUTER_MARGIN_W

    @property
    def right_x0(self) -> float:
        return OUTER_MARGIN_W + self.base_w + GAP_W

    @property
    def world_w(self) -> float:
        return OUTER_MARGIN_W * 2 + self.base_w * 2 + GAP_W

    @property
    def gap_x0(self) -> float:
        return self.left_x0 + self.base_w

    @property
    def gap_x1(self) -> float:
        return self.right_x0

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
BASE = (46, 133, 87)
BASE_TOP = (85, 188, 128)
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
BRICK_X_PHASE_CELLS: Dict[str, float] = {
    # These Rhino contours start half a cell out of phase with the common
    # construction grid. Without this correction, their real interlock
    # positions land on half-step wx values and are unreachable by the
    # integer-step placer.
    "A": 0.0,
    "E": 0.0,
}


def snap_to(value: float, step: float) -> float:
    return round(value / step) * step


def normalize_poly(
    coords_mm: Sequence[Tuple[float, float]],
    x_phase_cells: float = 0.0,
    kind: str = "brick",
) -> Tuple[Tuple[float, float], ...]:
    min_x = min(x for x, _y in coords_mm)
    min_y = min(y for _x, y in coords_mm)
    raw = [(snap_to((x - min_x) / CELL_MM, X_GRID), (y - min_y) / CELL_MM) for x, y in coords_mm]
    if kind == "base":
        positive_z = [z for _x, z in raw if z > EDGE_TOL]
        low_top = min(positive_z, default=0.0)
        z_levels = (0.0, low_top, low_top + HEX_RISE)
        pts = tuple(
            (x + x_phase_cells, min(z_levels, key=lambda level: abs(level - z)))
            for x, z in raw
        )
    else:
        pts = tuple(
            (x + x_phase_cells, snap_to(z, HEX_RISE))
            for x, z in raw
        )
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


def signed_area(poly: Sequence[Tuple[float, float]]) -> float:
    return sum(
        poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]
        for i in range(len(poly) - 1)
    ) / 2.0


@dataclass(frozen=True)
class BrickType:
    key: str
    poly: Tuple[Tuple[float, float], ...]
    top_edges: Tuple[Tuple[float, float, float], ...]
    bottom_edges: Tuple[Tuple[float, float, float], ...]
    x_min: float
    x_max: float
    width: float
    height: float
    cx: float
    cz: float
    mass: float
    base_key: str = ""
    flipped: bool = False


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
    ccw = signed_area(poly) > 0.0
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        if abs(z0 - z1) > EDGE_TOL or abs(x0 - x1) < EDGE_TOL:
            continue

        # For a CCW polygon, the filled material lies to the left of each
        # directed edge. A left-to-right horizontal edge therefore has material
        # above it (bottom contact face); a right-to-left edge has material
        # below it (top support face). Reverse for CW contours.
        dx = x1 - x0
        is_bottom = dx > 0 if ccw else dx < 0
        if want == "bottom" and not is_bottom:
            continue
        if want == "top" and is_bottom:
            continue
        edges.append((min(x0, x1), max(x0, x1), (z0 + z1) / 2))
    return tuple(sorted(edges))


def make_brick(key: str, coords_mm: Sequence[Tuple[float, float]]) -> BrickType:
    # Brick x-origin is a grid phase, not the leftmost vertex. Most contours
    # need a -0.5 cell offset so integer wx aligns contact faces to the shared
    # pitch; A is corrected through BRICK_X_PHASE_CELLS above.
    poly = normalize_poly(coords_mm, x_phase_cells=BRICK_X_PHASE_CELLS.get(key, -0.5), kind="brick")
    cx, cz, area = centroid_area(poly)
    x_min = min(x for x, _z in poly)
    x_max = max(x for x, _z in poly)
    width = x_max - x_min
    height = max(z for _x, z in poly)
    top_edges = horizontal_edges(poly, cz, "top")
    bottom_edges = horizontal_edges(poly, cz, "bottom")
    return BrickType(key, poly, top_edges, bottom_edges, x_min, x_max, width, height, cx, cz, area, key, False)


def flipped_poly(poly: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    pts = list(poly[:-1] if poly[0] == poly[-1] else poly)
    x_min = min(x for x, _z in pts)
    x_max = max(x for x, _z in pts)
    z_min = min(z for _x, z in pts)
    z_max = max(z for _x, z in pts)
    flipped = tuple((x_min + x_max - x, z_min + z_max - z) for x, z in pts)
    if flipped[0] != flipped[-1]:
        flipped = flipped + (flipped[0],)
    return flipped


def make_flipped_brick(source: BrickType) -> BrickType:
    key = f"{source.key}'"
    poly = flipped_poly(source.poly)
    cx, cz, area = centroid_area(poly)
    x_min = min(x for x, _z in poly)
    x_max = max(x for x, _z in poly)
    width = x_max - x_min
    height = max(z for _x, z in poly)
    top_edges = horizontal_edges(poly, cz, "top")
    bottom_edges = horizontal_edges(poly, cz, "bottom")
    return BrickType(key, poly, top_edges, bottom_edges, x_min, x_max, width, height, cx, cz, area, source.base_key, True)


def make_base(span: int, coords_mm: Sequence[Tuple[float, float]]) -> BaseType:
    poly = normalize_poly(coords_mm, kind="base")
    cx, cz, _area = centroid_area(poly)
    width = max(x for x, _z in poly)
    height = max(z for _x, z in poly)
    return BaseType(span, poly, horizontal_edges(poly, cz, "top"), width, height)


PRIMARY_BTYPES: Dict[str, BrickType] = {k: make_brick(k, v) for k, v in BRICK_COORDS_MM.items()}
BTYPES: Dict[str, BrickType] = dict(PRIMARY_BTYPES)
for _key in ("A", "B", "C", "E", "F"):
    _flipped = make_flipped_brick(PRIMARY_BTYPES[_key])
    BTYPES[_flipped.key] = _flipped
BRICK_LIBRARY_KEYS: Tuple[str, ...] = ("A", "A'", "B", "B'", "C", "C'", "D", "E", "E'", "F", "F'")
BASE_TYPES: Dict[int, BaseType] = {k: make_base(k, v) for k, v in BASE_COORDS_MM.items()}


def brick_color_key(bt_or_key) -> str:
    key = bt_or_key.key if isinstance(bt_or_key, BrickType) else str(bt_or_key)
    return key[0].upper()


def toggle_flip_key(key: str) -> str:
    if key.startswith("D"):
        return "D"
    base = key[0].upper()
    flipped = f"{base}'"
    return base if key.endswith("'") else (flipped if flipped in BTYPES else base)


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
        return self.wx + self.btype.x_min

    @property
    def x1(self) -> float:
        return self.wx + self.btype.x_max

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


def _edge_lr(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    if p0[0] <= p1[0]:
        return p0[0], p0[1], p1[0], p1[1]
    return p1[0], p1[1], p0[0], p0[1]


def _edge_key(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[int, int, int, int]:
    x0, z0, x1, z1 = _edge_lr(p0, p1)
    return (
        round(x0 * 2),
        round(z0 / HEX_RISE * 1000),
        round(x1 * 2),
        round(z1 / HEX_RISE * 1000),
    )


def _cell_key(x0: float, z0: float) -> Tuple[int, int]:
    return (round(x0 * 2), round(z0 / HEX_RISE * 1000))


def hex_cell_edges(
    x0: float,
    x1: float,
    z0: float,
) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    h = HEX_RISE
    return [
        ("lower", (x0 - 0.5, z0 + h), (x0, z0)),
        ("lower", (x0, z0), (x1, z0)),
        ("lower", (x1, z0), (x1 + 0.5, z0 + h)),
        ("upper", (x0 - 0.5, z0 + h), (x0, z0 + 2.0 * h)),
        ("upper", (x0, z0 + 2.0 * h), (x1, z0 + 2.0 * h)),
        ("upper", (x1, z0 + 2.0 * h), (x1 + 0.5, z0 + h)),
    ]


def brick_cell_edges(
    bt: BrickType,
    wx: float,
    wz: float,
    want: str,
    owner: int,
) -> List[Tuple[float, float, float, float, int]]:
    counts: Dict[Tuple[int, int, int, int], int] = {}
    entries: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []
    for x0, x1, z0 in bt.bottom_edges:
        for kind, p0, p1 in hex_cell_edges(wx + x0, wx + x1, wz + z0):
            entries.append((kind, p0, p1))
            key = _edge_key(p0, p1)
            counts[key] = counts.get(key, 0) + 1
    edges: List[Tuple[float, float, float, float, int]] = []
    for kind, p0, p1 in entries:
        if kind != want or counts[_edge_key(p0, p1)] != 1:
            continue
        edges.append((*_edge_lr(p0, p1), owner))
    return edges


def brick_cell_keys(bt: BrickType, wx: float, wz: float) -> set[Tuple[int, int]]:
    return {_cell_key(wx + x0, wz + z0) for x0, _x1, z0 in bt.bottom_edges}


def poly_contact_edges(
    poly: Sequence[Tuple[float, float]],
    want: str,
    owner: int,
) -> List[Tuple[float, float, float, float, int]]:
    ccw = signed_area(poly) > 0.0
    edges: List[Tuple[float, float, float, float, int]] = []
    for p0, p1 in poly_edges(poly):
        dx = p1[0] - p0[0]
        if abs(dx) <= EDGE_TOL:
            continue
        is_lower = dx > 0 if ccw else dx < 0
        if want == "lower" and not is_lower:
            continue
        if want == "upper" and is_lower:
            continue
        edges.append((*_edge_lr(p0, p1), owner))
    return edges


def lattice_support_edges(existing: Sequence["BrickInst"]) -> List[Tuple[float, float, float, float, int]]:
    left_base, right_base = base_world_polys()
    edges = poly_contact_edges(left_base, "upper", BASE_LEFT)
    edges.extend(poly_contact_edges(right_base, "upper", BASE_RIGHT))
    for b in existing:
        edges.extend(brick_cell_edges(b.btype, b.wx, b.wz, "upper", b.id))
    return edges


def lattice_occupied_keys(existing: Sequence["BrickInst"]) -> set[Tuple[int, int]]:
    occupied: set[Tuple[int, int]] = set()
    for b in existing:
        occupied.update(brick_cell_keys(b.btype, b.wx, b.wz))
    return occupied


def _edges_coincident(
    a: Tuple[float, float, float, float, int],
    b: Tuple[float, float, float, float, int],
    eps: float = CONTACT_TOL,
) -> bool:
    return (
        abs(a[0] - b[0]) <= eps
        and abs(a[1] - b[1]) <= eps
        and abs(a[2] - b[2]) <= eps
        and abs(a[3] - b[3]) <= eps
    )


def lattice_contacts(
    bt: BrickType,
    wx: float,
    wz: float,
    support_edges: Sequence[Tuple[float, float, float, float, int]],
) -> List[SurfaceSeg]:
    contacts: List[SurfaceSeg] = []
    for lower in brick_cell_edges(bt, wx, wz, "lower", 0):
        for upper in support_edges:
            if not _edges_coincident(lower, upper):
                continue
            x0 = min(lower[0], lower[2])
            x1 = max(lower[0], lower[2])
            contacts.append(SurfaceSeg(x0, x1, (lower[1] + lower[3]) / 2.0, upper[4]))
    return merge_contacts(contacts)


def compute_lattice_drops(
    bt: BrickType,
    wx: float,
    existing: Sequence["BrickInst"],
    max_results: int = 64,
) -> List[Tuple[float, List[SurfaceSeg]]]:
    support_edges = lattice_support_edges(existing)
    if not support_edges:
        return []
    occupied = lattice_occupied_keys(existing)
    local_lower_edges = brick_cell_edges(bt, 0.0, 0.0, "lower", 0)

    candidates: List[float] = []

    def add_candidate(value: float) -> None:
        if not any(abs(value - old) <= 1e-7 for old in candidates):
            candidates.append(value)

    for lx0, lz0, lx1, lz1, _owner in local_lower_edges:
        wx0 = wx + lx0
        wx1 = wx + lx1
        for sx0, sz0, sx1, sz1, _sowner in support_edges:
            if abs(wx0 - sx0) > CONTACT_TOL or abs(wx1 - sx1) > CONTACT_TOL:
                continue
            wz0 = sz0 - lz0
            if abs((wz0 + lz1) - sz1) <= CONTACT_TOL:
                add_candidate(wz0)

    if not candidates:
        return []

    base_polys = base_world_polys()
    legal: List[Tuple[float, List[SurfaceSeg]]] = []
    for wz in sorted(candidates, reverse=True):
        if brick_cell_keys(bt, wx, wz) & occupied:
            continue
        candidate_poly = world_poly(bt, wx, wz)
        if any(polys_overlap_area(candidate_poly, base_poly) for base_poly in base_polys):
            continue
        contacts = lattice_contacts(bt, wx, wz, support_edges)
        if contact_width(contacts) >= 0.2:
            legal.append((wz, contacts))
    return _select_drop_options(legal, max_results)


def base_world_polys() -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
    base = LAYOUT.base_shape
    return (
        tuple((LAYOUT.left_x0 + x, z) for x, z in base.poly),
        tuple((LAYOUT.right_x0 + x, z) for x, z in base.poly),
    )


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
    if signed_area(pts + [pts[0]]) < 0:
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


def support_polys(existing: Sequence["BrickInst"]) -> List[Tuple[int, Tuple[Tuple[float, float], ...]]]:
    base = LAYOUT.base_shape
    supports = [
        (BASE_LEFT, tuple((LAYOUT.left_x0 + x, z) for x, z in base.poly)),
        (BASE_RIGHT, tuple((LAYOUT.right_x0 + x, z) for x, z in base.poly)),
    ]
    supports.extend((b.id, world_poly(b.btype, b.wx, b.wz)) for b in existing)
    return supports


def poly_edges(poly: Sequence[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    return [(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]


def segment_z_at(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    x: float,
) -> Optional[float]:
    x0, z0 = p0
    x1, z1 = p1
    if abs(x1 - x0) <= 1e-9:
        return None
    if x < min(x0, x1) - ALIGN_TOL or x > max(x0, x1) + ALIGN_TOL:
        return None
    t = (x - x0) / (x1 - x0)
    return z0 + t * (z1 - z0)


def edge_slope(p0: Tuple[float, float], p1: Tuple[float, float]) -> Optional[float]:
    dx = p1[0] - p0[0]
    if abs(dx) <= EDGE_TOL:
        return None
    return (p1[1] - p0[1]) / dx


def nearly_equal(a: float, b: float, eps: float = ALIGN_TOL) -> bool:
    return abs(a - b) <= eps


def complete_edge_contact(
    hit0: float,
    hit1: float,
    ax0: float,
    ax1: float,
    bx0: float,
    bx1: float,
    eps: float = ALIGN_TOL,
) -> bool:
    upper_full = nearly_equal(hit0, ax0, eps) and nearly_equal(hit1, ax1, eps)
    lower_full = nearly_equal(hit0, bx0, eps) and nearly_equal(hit1, bx1, eps)
    return upper_full or lower_full


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

    exact: List[SurfaceSeg] = []
    for up0, up1 in poly_edges(upper):
        uslope = edge_slope(up0, up1)
        if uslope is None:
            continue
        ux0 = min(up0[0], up1[0])
        ux1 = max(up0[0], up1[0])
        for lo0, lo1 in poly_edges(lower):
            lslope = edge_slope(lo0, lo1)
            if lslope is None or abs(uslope - lslope) > 0.02:
                continue
            lx0 = min(lo0[0], lo1[0])
            lx1 = max(lo0[0], lo1[0])
            hit0 = max(ux0, min(lo0[0], lo1[0]))
            hit1 = min(ux1, max(lo0[0], lo1[0]))
            if hit1 - hit0 <= 0.05:
                continue
            if not complete_edge_contact(hit0, hit1, ux0, ux1, lx0, lx1):
                continue
            mid = (hit0 + hit1) / 2
            uz = segment_z_at(up0, up1, mid)
            lz = segment_z_at(lo0, lo1, mid)
            if uz is None or lz is None or abs(uz - lz) > eps:
                continue
            upper_bottom = any(abs(ub - uz) <= eps for ub, _ut in _vertical_fill_intervals(upper, mid))
            lower_top = any(abs(lt - lz) <= eps for _lb, lt in _vertical_fill_intervals(lower, mid))
            if not upper_bottom or not lower_top:
                continue
            exact.append(SurfaceSeg(hit0, hit1, (uz + lz) / 2, owner))
    if exact:
        return merge_contacts(exact)

    # Contact is only legal when a real edge of the candidate and a real edge
    # of the support are collinear and coincident over a visible segment. The
    # old sampled fallback could turn near-misses or half-step touches into
    # support, which made illegal half-brick placements appear valid.
    return []


def _raw_boundary_contact_segments(
    upper: Sequence[Tuple[float, float]],
    lower: Sequence[Tuple[float, float]],
    owner: int,
    eps: float = CONTACT_TOL,
) -> List[Tuple[SurfaceSeg, float, float]]:
    raw: List[Tuple[SurfaceSeg, float, float]] = []
    for up0, up1 in poly_edges(upper):
        uslope = edge_slope(up0, up1)
        if uslope is None:
            continue
        ux0 = min(up0[0], up1[0])
        ux1 = max(up0[0], up1[0])
        for lo0, lo1 in poly_edges(lower):
            lslope = edge_slope(lo0, lo1)
            if lslope is None or abs(uslope - lslope) > 0.02:
                continue
            lx0 = min(lo0[0], lo1[0])
            lx1 = max(lo0[0], lo1[0])
            hit0 = max(ux0, lx0)
            hit1 = min(ux1, lx1)
            if hit1 - hit0 <= 0.05:
                continue
            mid = (hit0 + hit1) / 2
            uz = segment_z_at(up0, up1, mid)
            lz = segment_z_at(lo0, lo1, mid)
            if uz is None or lz is None or abs(uz - lz) > eps:
                continue
            upper_bottom = any(abs(ub - uz) <= eps for ub, _ut in _vertical_fill_intervals(upper, mid))
            lower_top = any(abs(lt - lz) <= eps for _lb, lt in _vertical_fill_intervals(lower, mid))
            if not upper_bottom or not lower_top:
                continue
            raw.append((SurfaceSeg(hit0, hit1, (uz + lz) / 2, owner), ux0, ux1))
    return raw


def _covered_by_segments(x0: float, x1: float, segments: Sequence[SurfaceSeg]) -> bool:
    cursor = x0
    for seg in sorted(segments, key=lambda s: s.x0):
        if seg.x1 <= cursor + ALIGN_TOL:
            continue
        if seg.x0 > cursor + ALIGN_TOL:
            return False
        cursor = max(cursor, seg.x1)
        if cursor >= x1 - ALIGN_TOL:
            return True
    return cursor >= x1 - ALIGN_TOL


def multi_segment_candidate_contacts(
    raw: Sequence[Tuple[SurfaceSeg, float, float]],
) -> List[SurfaceSeg]:
    if len(raw) < 2:
        return []
    pieces = [seg for seg, _ex0, _ex1 in raw]
    total_width = sum(max(0.0, seg.x1 - seg.x0) for seg in pieces)
    span = max(seg.x1 for seg in pieces) - min(seg.x0 for seg in pieces)
    touched_edges = {
        (round(ex0, 4), round(ex1, 4), round(seg.z, 4))
        for seg, ex0, ex1 in raw
    }
    support_owners = {seg.owner for seg in pieces}
    if len(touched_edges) < 2 and len(support_owners) < 2:
        return []
    if total_width < 0.45 or span < 0.75:
        return []
    return merge_contacts(pieces)


def complete_candidate_edge_contacts(
    candidate_poly: Sequence[Tuple[float, float]],
    supports: Sequence[Tuple[int, Tuple[Tuple[float, float], ...]]],
) -> List[SurfaceSeg]:
    raw: List[Tuple[SurfaceSeg, float, float]] = []
    for owner, spoly in supports:
        raw.extend(_raw_boundary_contact_segments(candidate_poly, spoly, owner))

    contacts: List[SurfaceSeg] = []
    for up0, up1 in poly_edges(candidate_poly):
        uslope = edge_slope(up0, up1)
        if uslope is None:
            continue
        ux0 = min(up0[0], up1[0])
        ux1 = max(up0[0], up1[0])
        mid = (ux0 + ux1) / 2
        uz = segment_z_at(up0, up1, mid)
        if uz is None:
            continue
        upper_bottom = any(abs(ub - uz) <= CONTACT_TOL for ub, _ut in _vertical_fill_intervals(candidate_poly, mid))
        if not upper_bottom:
            continue
        pieces = [
            seg
            for seg, ex0, ex1 in raw
            if nearly_equal(ex0, ux0, CONTACT_TOL)
            and nearly_equal(ex1, ux1, CONTACT_TOL)
            and abs(seg.z - uz) <= CONTACT_TOL
        ]
        if pieces and _covered_by_segments(ux0, ux1, pieces):
            contacts.extend(pieces)
    # A placement can be supported by one complete bottom edge plus several
    # smaller interlocking contacts on other bricks. Returning as soon as a
    # complete edge is found hides those extra supports from the stability
    # check, so keep both complete-edge and multi-segment contacts.
    contacts = merge_contacts(list(contacts) + multi_segment_candidate_contacts(raw))
    return contacts


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

    return sampled_overlap_area(a, b, eps) > OVERLAP_AREA_TOL


def material_overlaps_existing(bt: BrickType, wx: float, wz: float, existing: Sequence["BrickInst"]) -> bool:
    candidate_poly = world_poly(bt, wx, wz)
    return any(
        polys_overlap_area(candidate_poly, world_poly(other.btype, other.wx, other.wz))
        for other in existing
    )


def vertical_drop_path_blocked(
    bt: BrickType,
    wx: float,
    wz: float,
    existing: Sequence[BrickInst],
    max_interlock_layers: int = 3,
) -> bool:
    if not existing:
        return False
    layer_height = 2.0 * HEX_RISE
    max_interlock_depth = max_interlock_layers * layer_height + ALIGN_TOL
    candidate_top = wz + bt.height
    candidate_x0 = wx + bt.x_min
    candidate_x1 = wx + bt.x_max
    highest_overlap_top = -math.inf
    for other in existing:
        if not overlap(candidate_x0, candidate_x1, other.x0, other.x1):
            continue
        highest_overlap_top = max(highest_overlap_top, other.wz + other.btype.height)
    return highest_overlap_top > candidate_top + max_interlock_depth


def local_bottom_at(bt: BrickType, lx: float) -> Optional[float]:
    for x0, x1, bz in bt.bottom_edges:
        if x0 - 1e-9 <= lx <= x1 + 1e-9:
            return bz
    return None


def compute_drop(bt: BrickType, wx: float, surfaces: Sequence[SurfaceSeg]) -> Tuple[Optional[float], List[SurfaceSeg]]:
    candidates: List[float] = []

    for sx in surfaces:
        for bx0, bx1, bbz in bt.bottom_edges:
            wx0 = wx + bx0
            wx1 = wx + bx1
            if not overlap(wx0, wx1, sx.x0, sx.x1):
                continue
            cand = sx.z - bbz
            if not any(abs(cand - old) <= 1e-9 for old in candidates):
                candidates.append(cand)

    if not candidates:
        return None, []

    for cand_wz in sorted(candidates, reverse=True):
        contacts: List[SurfaceSeg] = []
        for bx0, bx1, bbz in bt.bottom_edges:
            wx0 = wx + bx0
            wx1 = wx + bx1
            z = cand_wz + bbz
            covering = [
                sx for sx in surfaces
                if abs(sx.z - z) <= ALIGN_TOL and overlap(wx0, wx1, sx.x0, sx.x1)
            ]
            if not covering:
                continue
            cover_start = wx0
            edge_contacts: List[SurfaceSeg] = []
            for sx in sorted(covering, key=lambda s: s.x0):
                hit = overlap(cover_start, wx1, sx.x0, sx.x1)
                if not hit:
                    continue
                if hit[0] > cover_start + ALIGN_TOL:
                    break
                edge_contacts.append(SurfaceSeg(max(wx0, sx.x0), min(wx1, sx.x1), sx.z, sx.owner))
                cover_start = max(cover_start, sx.x1)
                if cover_start >= wx1 - ALIGN_TOL:
                    break
            if cover_start >= wx1 - ALIGN_TOL:
                contacts.extend(edge_contacts)

        if contacts:
            return cand_wz, merge_contacts(contacts)

    return None, []


def _select_drop_options(
    drops: Sequence[Tuple[float, List[SurfaceSeg]]],
    max_results: int,
) -> List[Tuple[float, List[SurfaceSeg]]]:
    if not drops:
        return []

    selected: List[Tuple[float, List[SurfaceSeg]]] = []
    per_bucket = max(1, max_results // 3)

    def add(drop: Tuple[float, List[SurfaceSeg]]) -> None:
        if len(selected) >= max_results:
            return
        if not any(abs(drop[0] - old[0]) <= ALIGN_TOL for old in selected):
            selected.append(drop)

    # Keep representative legal placements from the top, bottom, and broadest
    # contacts. A pure top-down cutoff misses legitimate lower interlocks once
    # the structure becomes crowded.
    for drop in sorted(drops, key=lambda d: (-d[0], -contact_width(d[1])))[:per_bucket]:
        add(drop)
    for drop in sorted(drops, key=lambda d: (d[0], -contact_width(d[1])))[:per_bucket]:
        add(drop)
    for drop in sorted(drops, key=lambda d: (-contact_width(d[1]), -d[0]))[: max_results - len(selected)]:
        add(drop)

    if len(selected) < max_results:
        for drop in sorted(drops, key=lambda d: (-d[0], -contact_width(d[1]))):
            add(drop)
            if len(selected) >= max_results:
                break

    selected.sort(key=lambda d: (-d[0], -contact_width(d[1])))
    return selected


def compute_geometric_drops(
    bt: BrickType,
    wx: float,
    surfaces: Sequence[SurfaceSeg],
    supports: Sequence[Tuple[int, Tuple[Tuple[float, float], ...]]],
    max_results: int = 64,
    complete_edges: bool = True,
) -> List[Tuple[float, List[SurfaceSeg]]]:
    candidates: List[float] = []

    def add_candidate(value: float) -> None:
        if not any(abs(value - old) <= 1e-7 for old in candidates):
            candidates.append(value)

    for sx in surfaces:
        for bx0, bx1, bbz in bt.bottom_edges:
            if overlap(wx + bx0, wx + bx1, sx.x0, sx.x1):
                add_candidate(sx.z - bbz)

    for _owner, spoly in supports:
        for bx, bz in bt.poly[:-1]:
            bwx = wx + bx
            for sx, sz in spoly[:-1]:
                if abs(bwx - sx) <= ALIGN_TOL:
                    add_candidate(sz - bz)

    brick_edges = poly_edges(bt.poly)
    for _owner, spoly in supports:
        for bp0, bp1 in brick_edges:
            bslope = edge_slope(bp0, bp1)
            if bslope is None:
                continue
            bx0 = wx + min(bp0[0], bp1[0])
            bx1 = wx + max(bp0[0], bp1[0])
            for sp0, sp1 in poly_edges(spoly):
                sslope = edge_slope(sp0, sp1)
                if sslope is None or abs(bslope - sslope) > 0.02:
                    continue
                x0 = max(bx0, min(sp0[0], sp1[0]))
                x1 = min(bx1, max(sp0[0], sp1[0]))
                if x1 - x0 <= 0.20:
                    continue
                x = (x0 + x1) / 2
                bz = segment_z_at(bp0, bp1, x - wx)
                sz = segment_z_at(sp0, sp1, x)
                if bz is not None and sz is not None:
                    add_candidate(sz - bz)

    if candidates:
        # A valid interlock can sit one or more vertical pitches below the
        # first edge/vertex alignment, especially when a lower brick creates a
        # recess. Add downward-only "settled" heights and let the final
        # overlap/contact/path tests decide legality.
        seeds = list(candidates)
        vertical_step = HEX_RISE / 2.0
        min_i = -12 if complete_edges else -16
        max_i = 0 if complete_edges else 16
        for seed in seeds:
            for i in range(min_i, max_i + 1):
                add_candidate(seed + i * vertical_step)

    if not candidates:
        return []

    legal: List[Tuple[float, List[SurfaceSeg]]] = []
    for wz in sorted(candidates, reverse=True):
        candidate_poly = world_poly(bt, wx, wz)
        blocked = False
        for owner, spoly in supports:
            if polys_overlap_area(candidate_poly, spoly):
                blocked = True
                break
        if blocked:
            continue
        if complete_edges:
            contacts = complete_candidate_edge_contacts(candidate_poly, supports)
        else:
            contacts = merge_contacts(
                [
                seg
                for owner, spoly in supports
                for seg, _ex0, _ex1 in _raw_boundary_contact_segments(candidate_poly, spoly, owner)
                ]
            )
        if contact_width(contacts) >= 0.2:
            legal.append((wz, contacts))

    if not complete_edges:
        legal.sort(key=lambda d: (-d[0], -contact_width(d[1])))
        return legal[:max_results]
    return _select_drop_options(legal, max_results)


def compute_geometric_drop(
    bt: BrickType,
    wx: float,
    surfaces: Sequence[SurfaceSeg],
    supports: Sequence[Tuple[int, Tuple[Tuple[float, float], ...]]],
) -> Tuple[Optional[float], List[SurfaceSeg]]:
    drops = compute_geometric_drops(bt, wx, surfaces, supports, max_results=1)
    if not drops:
        return None, []
    return drops[0]


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


def copy_contacts(contacts: Sequence[SurfaceSeg]) -> List[SurfaceSeg]:
    return [SurfaceSeg(c.x0, c.x1, c.z, c.owner) for c in contacts]


def copy_rec(rec: Rec) -> Rec:
    return Rec(
        rec.side,
        rec.bt,
        rec.wx,
        rec.wz,
        copy_contacts(rec.contacts),
        rec.score,
        rec.margin_after,
        rec.dm,
        rec.reach_gain,
        rec.intent,
    )


def bridge_state_key(bridge: "Bridge") -> Tuple:
    return (
        LAYOUT.base_span,
        tuple(
            (
                b.btype.key,
                round(b.wx, 4),
                round(b.wz, 4),
                b.actor,
            )
            for b in bridge.bricks
        ),
    )


def cache_put(cache: Dict, key: Tuple, value) -> None:
    if len(cache) > CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def log_event(tag: str, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{stamp}] [{tag}] {message}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def fmt_contacts(contacts: Sequence[SurfaceSeg]) -> str:
    if not contacts:
        return "[]"
    return "[" + ", ".join(
        f"({c.x0:.2f}-{c.x1:.2f}@{c.z:.2f}, owner={c.owner})"
        for c in contacts
    ) + "]"


def fmt_brick(b: Optional[BrickInst]) -> str:
    if b is None:
        return "None"
    return (
        f"{b.btype.key}#{b.id} actor={b.actor} "
        f"wx={b.wx:.2f} wz={b.wz:.2f} x=({b.x0:.2f},{b.x1:.2f}) "
        f"contacts={fmt_contacts(b.contacts)}"
    )


def fmt_state(bridge: "Bridge") -> str:
    mm = "inf" if bridge.min_margin == math.inf else f"{bridge.min_margin:.3f}"
    return (
        f"bricks={len(bridge.bricks)}/{MAX_BRICKS} margin={mm} "
        f"closed={bridge.bridge_closed} success={bridge.bridge_succeeded} "
        f"failed={bridge.brick_exhausted} focus={active_user_side(bridge) or 'both'}"
    )


def log_recommendations(recs: Sequence[Rec], label: str = "recs") -> None:
    if not recs:
        log_event("RECS", f"{label}: none")
        return
    rows = []
    for r in recs[:3]:
        rows.append(
            f"{r.side}:{r.bt.key} wx={r.wx:.2f} wz={r.wz:.2f} "
            f"score={r.score:.2f} margin={r.margin_after:.2f} intent={r.intent}"
        )
    log_event("RECS", f"{label}: " + " | ".join(rows))


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

    def contacts_bridge_sides(self, contacts: Sequence[SurfaceSeg]) -> bool:
        left_component = self.component_from(BASE_LEFT)
        right_component = self.component_from(BASE_RIGHT)
        owners = {c.owner for c in contacts}
        return bool(owners & left_component) and bool(owners & right_component)

    def owner_top_surfaces(self, owner: int) -> List[SurfaceSeg]:
        if owner in (BASE_LEFT, BASE_RIGHT):
            return [s for s in base_surfaces() if s.owner == owner]
        for b in self.bricks:
            if b.id == owner:
                return top_surfaces(b)
        return []

    def is_horizontal_top_support_contact(
        self,
        bt: BrickType,
        wx: float,
        wz: float,
        contact: SurfaceSeg,
    ) -> bool:
        candidate_bottom = False
        for x0, x1, z in bt.bottom_edges:
            hit = overlap(wx + x0, wx + x1, contact.x0, contact.x1)
            if hit and hit[1] - hit[0] >= 0.20 and abs(wz + z - contact.z) <= CONTACT_TOL:
                candidate_bottom = True
                break
        if not candidate_bottom:
            return False
        for surface in self.owner_top_surfaces(contact.owner):
            hit = overlap(surface.x0, surface.x1, contact.x0, contact.x1)
            if hit and hit[1] - hit[0] >= 0.20 and abs(surface.z - contact.z) <= CONTACT_TOL:
                return True
        return False

    def horizontal_contacts_bridge_sides(
        self,
        bt: BrickType,
        wx: float,
        wz: float,
        contacts: Sequence[SurfaceSeg],
        exclude_id: Optional[int] = None,
    ) -> bool:
        left_component = self.component_from(BASE_LEFT, exclude_id=exclude_id)
        right_component = self.component_from(BASE_RIGHT, exclude_id=exclude_id)
        left_hit = False
        right_hit = False
        for contact in contacts:
            if not self.is_horizontal_top_support_contact(bt, wx, wz, contact):
                continue
            if contact.owner in left_component:
                left_hit = True
            if contact.owner in right_component:
                right_hit = True
        return left_hit and right_hit

    def is_closing_candidate(self, bt: BrickType, wx: float, wz: float, contacts: Sequence[SurfaceSeg]) -> bool:
        return (
            self.remaining_gap <= CLOSING_MAX_GAP_W + CONTACT_TOL
            and self.remaining_gap <= bt.width + CONTACT_TOL
            and self.horizontal_contacts_bridge_sides(bt, wx, wz, contacts)
        )

    def try_place_options(
        self,
        bt: BrickType,
        wx: float,
        actor: str,
        max_options: int = 64,
        complete_edges: bool = True,
    ) -> List[BrickInst]:
        if abs(wx - snap_place_x(wx)) > 1e-6:
            return []
        cache_key = ("place", PLACE_RULE_VERSION, bridge_state_key(self), bt.key, round(wx, 4), max_options, complete_edges)
        cached = PLACE_OPTIONS_CACHE.get(cache_key)
        if cached is not None:
            return [
                BrickInst(self.next_id, bt, wx, wz, actor, copy_contacts(contacts))
                for wz, contacts in cached
            ]

        lattice_drops = compute_lattice_drops(bt, wx, self.bricks, max_results=max_options)
        # Lattice drops are fast, but they can miss lower interlocks where one
        # brick is carried by several supports. Always supplement them with the
        # polygon contact solver and let the real overlap/support checks decide.
        geometric_drops: List[Tuple[float, List[SurfaceSeg]]] = []
        if not lattice_drops or actor == "human":
            geometric_drops = compute_geometric_drops(
                bt,
                wx,
                self.all_surfaces(),
                support_polys(self.bricks),
                max_results=max(max_options, 16),
                complete_edges=complete_edges,
            )
        merged_drops: List[Tuple[float, List[SurfaceSeg]]] = []
        for wz, contacts in list(lattice_drops) + list(geometric_drops):
            found = False
            for i, (old_wz, old_contacts) in enumerate(merged_drops):
                if abs(wz - old_wz) <= ALIGN_TOL:
                    merged_drops[i] = (old_wz, merge_contacts(list(old_contacts) + list(contacts)))
                    found = True
                    break
            if not found:
                merged_drops.append((wz, contacts))
        drops = _select_drop_options(merged_drops, max_options)
        options: List[BrickInst] = []
        for wz, contacts in drops:
            closes_bridge = self.is_closing_candidate(bt, wx, wz, contacts)
            if contact_width(contacts) < 0.2 and not closes_bridge:
                continue
            candidate_poly = world_poly(bt, wx, wz)
            if material_overlaps_existing(bt, wx, wz, self.bricks):
                continue
            if not closes_bridge and vertical_drop_path_blocked(bt, wx, wz, self.bricks):
                continue
            option = BrickInst(self.next_id, bt, wx, wz, actor, contacts)
            if not support_contact_ok(option, closes_bridge):
                continue
            options.append(option)
        cache_put(
            PLACE_OPTIONS_CACHE,
            cache_key,
            [(option.wz, copy_contacts(option.contacts)) for option in options],
        )
        return options

    def try_place(self, bt: BrickType, wx: float, actor: str, target_wz: Optional[float] = None) -> Optional[BrickInst]:
        options = self.try_place_options(bt, wx, actor)
        if target_wz is not None:
            for option in options:
                if abs(option.wz - target_wz) <= 1e-5:
                    return option
            return None
        return options[0] if options else None

    def place(self, bt: BrickType, wx: float, actor: str, target_wz: Optional[float] = None) -> Optional[BrickInst]:
        brick = self.try_place(bt, wx, actor, target_wz)
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

    def contact_graph(self, exclude_id: Optional[int] = None) -> Dict[int, set[int]]:
        nodes = {BASE_LEFT, BASE_RIGHT}
        nodes.update(b.id for b in self.bricks if b.id != exclude_id)
        graph: Dict[int, set[int]] = {node: set() for node in nodes}
        for b in self.bricks:
            if b.id == exclude_id:
                continue
            for owner in {c.owner for c in b.contacts}:
                if owner == exclude_id:
                    continue
                graph.setdefault(b.id, set()).add(owner)
                graph.setdefault(owner, set()).add(b.id)
        return graph

    def component_from(self, start: int, exclude_id: Optional[int] = None) -> set[int]:
        graph = self.contact_graph(exclude_id)
        seen: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph.get(node, set()) - seen)
        return seen

    def dual_base_supported_ids(self) -> set[int]:
        ids: set[int] = set()
        for b in self.bricks:
            if self.horizontal_contacts_bridge_sides(b.btype, b.wx, b.wz, b.contacts, exclude_id=b.id):
                ids.add(b.id)
        return ids

    @property
    def closing_brick_ids(self) -> set[int]:
        return self.dual_base_supported_ids()

    def is_closing_brick(self, bid: int) -> bool:
        return bid in self.closing_brick_ids

    @property
    def min_margin(self) -> float:
        return min((j.margin for j in self.joints), default=math.inf)

    @property
    def danger(self) -> float:
        if not self.joints:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.min_margin / 1.25))

    @property
    def left_front(self) -> float:
        return max(
            [LAYOUT.gap_x0]
            + [
                b.x1
                for b in self.bricks
                if (b.x0 + b.x1) / 2 < LAYOUT.center_x and b.x1 > LAYOUT.gap_x0
            ]
        )

    @property
    def right_front(self) -> float:
        return min(
            [LAYOUT.gap_x1]
            + [
                b.x0
                for b in self.bricks
                if (b.x0 + b.x1) / 2 >= LAYOUT.center_x and b.x0 < LAYOUT.gap_x1
            ]
        )

    @property
    def remaining_gap(self) -> float:
        return max(0.0, self.right_front - self.left_front)

    @property
    def left_reach(self) -> float:
        return max(0.0, self.left_front - LAYOUT.gap_x0)

    @property
    def right_reach(self) -> float:
        return max(0.0, LAYOUT.gap_x1 - self.right_front)

    @property
    def gap_coverage(self) -> float:
        return min(1.0, (self.left_reach + self.right_reach) / GAP_W)

    @property
    def bridge_closed(self) -> bool:
        return bool(self.closing_brick_ids)

    @property
    def structurally_stable(self) -> bool:
        return self.min_margin >= MIN_ACCEPT_MARGIN

    @property
    def bridge_succeeded(self) -> bool:
        return self.bridge_closed

    @property
    def brick_exhausted(self) -> bool:
        return len(self.bricks) >= MAX_BRICKS and not self.bridge_succeeded


def contact_width(contacts: Sequence[SurfaceSeg]) -> float:
    return sum(max(0.0, c.x1 - c.x0) for c in contacts)


def side_for_x(wx: float) -> str:
    return "L" if wx < LAYOUT.center_x else "R"


def active_user_side(bridge: Bridge) -> Optional[str]:
    for b in reversed(bridge.bricks):
        if b.actor == "human":
            return side_for_x((b.x0 + b.x1) / 2)
    return None


def reach_for_side(brick: BrickInst, side: str) -> float:
    if side == "L":
        return max(0.0, brick.x1 - LAYOUT.gap_x0)
    return max(0.0, LAYOUT.gap_x1 - brick.x0)


def reach_balance(bridge: Bridge) -> float:
    return abs(bridge.left_reach - bridge.right_reach)


def no_progress_streak(bridge: Bridge, eps: float = 0.05) -> int:
    replay = Bridge()
    streak = 0
    for b in bridge.bricks:
        before = replay.left_reach + replay.right_reach
        replay.bricks.append(
            BrickInst(b.id, b.btype, b.wx, b.wz, b.actor, list(b.contacts), b.margin)
        )
        replay.next_id = max(replay.next_id, b.id + 1)
        replay.analyse()
        after = replay.left_reach + replay.right_reach
        if after - before > eps:
            streak = 0
        else:
            streak += 1
    return streak


def base_contact_width(contacts: Sequence[SurfaceSeg]) -> float:
    return sum(max(0.0, c.x1 - c.x0) for c in contacts if c.owner < 0)


def union_width(intervals: Sequence[Tuple[float, float]]) -> float:
    valid = sorted((a, b) for a, b in intervals if b - a > 1e-6)
    if not valid:
        return 0.0
    total = 0.0
    cur0, cur1 = valid[0]
    for x0, x1 in valid[1:]:
        if x0 <= cur1 + 1e-6:
            cur1 = max(cur1, x1)
        else:
            total += cur1 - cur0
            cur0, cur1 = x0, x1
    total += cur1 - cur0
    return total


def merged_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid = sorted((a, b) for a, b in intervals if b - a > 1e-6)
    if not valid:
        return []
    merged: List[Tuple[float, float]] = []
    cur0, cur1 = valid[0]
    for x0, x1 in valid[1:]:
        if x0 <= cur1 + 1e-6:
            cur1 = max(cur1, x1)
        else:
            merged.append((cur0, cur1))
            cur0, cur1 = x0, x1
    merged.append((cur0, cur1))
    return merged


def low_foundation_intervals(bridge: Bridge, owner: int) -> List[Tuple[float, float]]:
    if owner == BASE_LEFT:
        span = (LAYOUT.left_x0, LAYOUT.left_x0 + LAYOUT.base_w)
    else:
        span = (LAYOUT.right_x0, LAYOUT.right_x0 + LAYOUT.base_w)
    intervals: List[Tuple[float, float]] = []
    for b in bridge.bricks:
        if b.wz > low_layer_limit():
            continue
        if not any(c.owner == owner for c in b.contacts):
            continue
        clipped = (max(span[0], b.x0), min(span[1], b.x1))
        if clipped[1] - clipped[0] > 1e-6:
            intervals.append(clipped)
    return merged_intervals(intervals)


def low_layer_limit() -> float:
    return LAYOUT.base_shape.height + LOW_LAYER_CLEARANCE


def foundation_coverage_by_owner(bridge: Bridge) -> Dict[int, float]:
    spans = {
        BASE_LEFT: (LAYOUT.left_x0, LAYOUT.left_x0 + LAYOUT.base_w),
        BASE_RIGHT: (LAYOUT.right_x0, LAYOUT.right_x0 + LAYOUT.base_w),
    }
    coverage: Dict[int, float] = {}
    for owner, (x0, x1) in spans.items():
        coverage[owner] = min(1.0, union_width(low_foundation_intervals(bridge, owner)) / max(1e-6, x1 - x0))
    return coverage


def owner_for_side(side: str) -> int:
    return BASE_LEFT if side == "L" else BASE_RIGHT


def foundation_coverage_for_side(bridge: Bridge, side: str) -> float:
    cov = foundation_coverage_by_owner(bridge)
    return cov.get(owner_for_side(side), 0.0)


def base_span_for_owner(owner: int) -> Tuple[float, float]:
    if owner == BASE_LEFT:
        return LAYOUT.left_x0, LAYOUT.left_x0 + LAYOUT.base_w
    return LAYOUT.right_x0, LAYOUT.right_x0 + LAYOUT.base_w


def base_cell_occupancy(bridge: Bridge, owner: int, cell: float = CELL) -> List[bool]:
    span0, span1 = base_span_for_owner(owner)
    count = max(1, int(math.ceil((span1 - span0) / cell)))
    occ = [False] * count
    for x0, x1 in low_foundation_intervals(bridge, owner):
        first = max(0, int(math.floor((x0 - span0) / cell)))
        last = min(count - 1, int(math.ceil((x1 - span0) / cell)) - 1)
        for i in range(first, last + 1):
            c0 = span0 + i * cell
            c1 = min(span1, c0 + cell)
            if min(x1, c1) - max(x0, c0) >= cell * 0.45:
                occ[i] = True
    return occ


def base_cell_utilization_for_side(bridge: Bridge, side: str) -> float:
    occ = base_cell_occupancy(bridge, owner_for_side(side))
    return sum(1 for filled in occ if filled) / max(1, len(occ))


def base_trapped_void_score_for_side(bridge: Bridge, side: str) -> float:
    occ = base_cell_occupancy(bridge, owner_for_side(side))
    if len(occ) < 3:
        return 0.0
    trapped = 0
    for i, filled in enumerate(occ):
        if filled:
            continue
        left_filled = any(occ[:i])
        right_filled = any(occ[i + 1 :])
        if left_filled and right_filled:
            trapped += 1
    return trapped / len(occ)


def foundation_coverage(bridge: Bridge) -> float:
    cov = foundation_coverage_by_owner(bridge)
    return min(cov.get(BASE_LEFT, 0.0), cov.get(BASE_RIGHT, 0.0))


def foundation_average_coverage(bridge: Bridge) -> float:
    cov = foundation_coverage_by_owner(bridge)
    return (cov.get(BASE_LEFT, 0.0) + cov.get(BASE_RIGHT, 0.0)) / 2.0


def foundation_target(strategy: str) -> float:
    if strategy == "conservative":
        return CONSERVATIVE_FOUNDATION_TARGET
    return AGGRESSIVE_FOUNDATION_TARGET


def foundation_stage_required(bridge: Bridge, strategy: str) -> bool:
    return foundation_coverage(bridge) < foundation_target(strategy)


def foundation_stage_required_for_side(bridge: Bridge, strategy: str, side: str) -> bool:
    return foundation_coverage_for_side(bridge, side) < foundation_target(strategy)


def is_foundation_move(bi: BrickInst) -> bool:
    return bi.wz <= low_layer_limit() and base_contact_width(bi.contacts) >= 0.2


def foundation_packing_score(bridge: Bridge) -> float:
    min_brick_width = min(bt.width for bt in BTYPES.values())
    spans = {
        BASE_LEFT: (LAYOUT.left_x0, LAYOUT.left_x0 + LAYOUT.base_w),
        BASE_RIGHT: (LAYOUT.right_x0, LAYOUT.right_x0 + LAYOUT.base_w),
    }
    total_score = 0.0
    for owner, (span0, span1) in spans.items():
        covered = low_foundation_intervals(bridge, owner)
        cursor = span0
        usable = 0.0
        unusable = 0.0
        fragments = 0
        for x0, x1 in covered:
            gap = max(0.0, x0 - cursor)
            if 0.15 < gap < min_brick_width - 0.15:
                unusable += gap
                fragments += 1
            elif gap >= min_brick_width - 0.15:
                usable += gap
            cursor = max(cursor, x1)
        gap = max(0.0, span1 - cursor)
        if 0.15 < gap < min_brick_width - 0.15:
            unusable += gap
            fragments += 1
        elif gap >= min_brick_width - 0.15:
            usable += gap
        total_score += max(
            0.0,
            1.0
            + 0.45 * usable / max(1e-6, span1 - span0)
            - 1.30 * unusable / max(1e-6, span1 - span0)
            - 0.12 * fragments,
        )
    return total_score / 2.0


def foundation_open_slot_score(bridge: Bridge) -> float:
    min_brick_width = min(bt.width for bt in BTYPES.values())
    spans = {
        BASE_LEFT: (LAYOUT.left_x0, LAYOUT.left_x0 + LAYOUT.base_w),
        BASE_RIGHT: (LAYOUT.right_x0, LAYOUT.right_x0 + LAYOUT.base_w),
    }
    active_scores: List[float] = []
    for owner, (span0, span1) in spans.items():
        covered = low_foundation_intervals(bridge, owner)
        if not covered:
            continue
        cursor = span0
        largest_gap = 0.0
        for x0, x1 in covered:
            largest_gap = max(largest_gap, max(0.0, x0 - cursor))
            cursor = max(cursor, x1)
        largest_gap = max(largest_gap, max(0.0, span1 - cursor))
        active_scores.append(min(1.0, largest_gap / max(1e-6, min_brick_width)))
    if not active_scores:
        return 1.0
    return sum(active_scores) / len(active_scores)


def foundation_support_efficiency(bi: BrickInst) -> float:
    return min(1.0, base_contact_width(bi.contacts) / max(1e-6, bi.btype.width))


def foundation_score(bridge: Bridge, bi: BrickInst) -> float:
    low_bonus = max(0.0, 4.0 - bi.wz) / 4.0
    base_bonus = min(1.0, base_contact_width(bi.contacts) / 4.0)
    early = max(0.0, 1.0 - bridge.gap_coverage)
    return early * (0.65 * base_bonus + 0.35 * low_bonus)


def normalized_margin(value: float) -> float:
    if value == math.inf:
        return 1.0
    return min(1.0, max(0.0, value))


def stable_enough_margin(value: float) -> bool:
    return value >= MIN_ACCEPT_MARGIN


def normalized_contact_width(bi: BrickInst) -> float:
    return min(1.0, contact_width(bi.contacts) / max(1e-6, bi.btype.width))


def support_contact_ok(bi: BrickInst, closes_bridge: bool = False) -> bool:
    if closes_bridge:
        return True
    total = contact_width(bi.contacts)
    if total < MIN_SUPPORT_CONTACT_WIDTH:
        owners = {c.owner for c in bi.contacts}
        brick_owners = {owner for owner in owners if owner > 0}
        if len(brick_owners) < 2:
            return False
        span = max(c.x1 for c in bi.contacts) - min(c.x0 for c in bi.contacts)
        ratio = total / max(1e-6, bi.btype.width)
        min_span = min(2.0, bi.btype.width * 0.30)
        if total < 1.0 or ratio < 0.16 or span < min_span:
            return False
        return True
    ratio = total / max(1e-6, bi.btype.width)
    if base_contact_width(bi.contacts) >= 0.2:
        return ratio >= MIN_BASE_CONTACT_RATIO
    return ratio >= MIN_SUPPORT_CONTACT_RATIO


def joint_reinforcement_score(before: "Bridge", candidate: BrickInst) -> float:
    contact_by_owner: Dict[int, float] = {}
    for c in candidate.contacts:
        if c.owner <= 0:
            continue
        contact_by_owner[c.owner] = contact_by_owner.get(c.owner, 0.0) + max(0.0, c.x1 - c.x0)
    if len(contact_by_owner) < 2:
        return 0.0

    by_id = {b.id: b for b in before.bricks}
    owners = [owner for owner, width in contact_by_owner.items() if width >= 0.45 and owner in by_id]
    if len(owners) < 2:
        return 0.0

    best = 0.0
    for i, left_id in enumerate(owners):
        for right_id in owners[i + 1 :]:
            a = by_id[left_id]
            b = by_id[right_id]
            if abs(layer_key(a.wz) - layer_key(b.wz)) > HEX_RISE + ALIGN_TOL:
                continue
            seam0 = min(a.x1, b.x1)
            seam1 = max(a.x0, b.x0)
            if seam1 < seam0:
                seam_x = (max(a.x0, b.x0) + min(a.x1, b.x1)) / 2.0
            else:
                seam_x = (seam0 + seam1) / 2.0
            if not (candidate.x0 - ALIGN_TOL <= seam_x <= candidate.x1 + ALIGN_TOL):
                continue
            pair_contact = contact_by_owner[left_id] + contact_by_owner[right_id]
            contact_score = min(1.0, pair_contact / max(1e-6, candidate.btype.width * 0.45))
            center_bonus = 1.0 - min(
                1.0,
                abs(candidate.world_cx - seam_x) / max(1e-6, candidate.btype.width * 0.5),
            )
            best = max(best, 0.75 * contact_score + 0.25 * center_bonus)
    return best


def _contact_width_by_owner(candidate: BrickInst) -> Dict[int, float]:
    widths: Dict[int, float] = {}
    for c in candidate.contacts:
        if c.owner <= 0:
            continue
        widths[c.owner] = widths.get(c.owner, 0.0) + max(0.0, c.x1 - c.x0)
    return widths


def _pair_seam_x(a: BrickInst, b: BrickInst) -> float:
    seam0 = min(a.x1, b.x1)
    seam1 = max(a.x0, b.x0)
    if seam1 < seam0:
        return (max(a.x0, b.x0) + min(a.x1, b.x1)) / 2.0
    return (seam0 + seam1) / 2.0


def pair_has_upper_lock(before: Bridge, a: BrickInst, b: BrickInst, seam_x: float) -> bool:
    for child in before.bricks:
        if child.id in (a.id, b.id):
            continue
        if child.wz <= max(a.wz, b.wz) + ALIGN_TOL:
            continue
        contact_by_owner = _contact_width_by_owner(child)
        if contact_by_owner.get(a.id, 0.0) < 0.35 or contact_by_owner.get(b.id, 0.0) < 0.35:
            continue
        if child.x0 - ALIGN_TOL <= seam_x <= child.x1 + ALIGN_TOL:
            return True
    return False


def exposed_cantilever_pairs(before: Bridge, side: str) -> List[Tuple[BrickInst, BrickInst, float, float]]:
    pairs: List[Tuple[BrickInst, BrickInst, float, float]] = []
    min_pair_reach = 1.2
    max_pair_gap = 2.2
    for z in side_layer_levels(before, side):
        layer = sorted(side_bricks_on_layer(before, side, z), key=lambda b: b.x0)
        for i, a in enumerate(layer):
            for b in layer[i + 1 :]:
                if b.x0 - a.x1 > max_pair_gap:
                    break
                pair_reach = max(reach_for_side(a, side), reach_for_side(b, side))
                if pair_reach < min_pair_reach:
                    continue
                seam_x = _pair_seam_x(a, b)
                if not (min(a.x0, b.x0) - ALIGN_TOL <= seam_x <= max(a.x1, b.x1) + ALIGN_TOL):
                    continue
                if pair_has_upper_lock(before, a, b, seam_x):
                    continue
                severity = min(1.0, pair_reach / 4.0)
                pairs.append((a, b, seam_x, severity))
    return pairs


def urgent_joint_reinforcement_score(before: Bridge, candidate: BrickInst, side: str) -> float:
    contact_by_owner = _contact_width_by_owner(candidate)
    if len(contact_by_owner) < 2:
        return 0.0
    best = 0.0
    for a, b, seam_x, severity in exposed_cantilever_pairs(before, side):
        if contact_by_owner.get(a.id, 0.0) < 0.35 or contact_by_owner.get(b.id, 0.0) < 0.35:
            continue
        if candidate.wz <= max(a.wz, b.wz) + ALIGN_TOL:
            continue
        if not (candidate.x0 - ALIGN_TOL <= seam_x <= candidate.x1 + ALIGN_TOL):
            continue
        pair_contact = contact_by_owner[a.id] + contact_by_owner[b.id]
        contact_score = min(1.0, pair_contact / max(1e-6, candidate.btype.width * 0.38))
        center_score = 1.0 - min(
            1.0,
            abs(candidate.world_cx - seam_x) / max(1e-6, candidate.btype.width * 0.45),
        )
        best = max(best, 0.50 * severity + 0.35 * contact_score + 0.15 * center_score)
    return best


def side_reach(bridge: Bridge, side: str) -> float:
    return bridge.left_reach if side == "L" else bridge.right_reach


def side_reach_gain(before: Bridge, after: Bridge, side: str) -> float:
    return max(0.0, side_reach(after, side) - side_reach(before, side))


def strategy_closes_bridge(before: Bridge, after: Bridge, candidate: BrickInst, strategy: str) -> bool:
    if not after.is_closing_brick(candidate.id):
        return False
    if strategy == "conservative":
        if candidate.btype.base_key == "F":
            return False
        return before.remaining_gap <= CONSERVATIVE_CLOSING_MAX_GAP_W + CONTACT_TOL
    return before.remaining_gap <= CLOSING_MAX_GAP_W + CONTACT_TOL


def layer_key(z: float) -> float:
    return round(z / max(1e-6, HEX_RISE / 2.0)) * (HEX_RISE / 2.0)


def same_layer(a: float, b: float) -> bool:
    return abs(layer_key(a) - layer_key(b)) <= ALIGN_TOL


def side_bricks_on_layer(bridge: Bridge, side: str, z: float) -> List[BrickInst]:
    return [
        b for b in bridge.bricks
        if side_for_x((b.x0 + b.x1) / 2.0) == side and same_layer(b.wz, z)
    ]


def is_behind_candidate(other: BrickInst, candidate: BrickInst, side: str) -> bool:
    if side == "L":
        return other.world_cx < candidate.world_cx - ALIGN_TOL
    return other.world_cx > candidate.world_cx + ALIGN_TOL


def aggressive_layer_constraint_ok(before: Bridge, candidate: BrickInst, side: str, reach_gain: float, closes_bridge: bool) -> bool:
    if closes_bridge or reach_gain <= 0.05:
        return True
    rear = [
        b for b in side_bricks_on_layer(before, side, candidate.wz)
        if is_behind_candidate(b, candidate, side)
    ]
    return len(rear) >= AGGRESSIVE_LAYER_REAR_BRICKS


def creates_unusable_foundation_island(before: Bridge, candidate: BrickInst, side: str) -> bool:
    if not is_foundation_move(candidate):
        return False
    owner = owner_for_side(side)
    span0, span1 = base_span_for_owner(owner)
    x0 = max(span0, candidate.x0)
    x1 = min(span1, candidate.x1)
    if x1 - x0 <= 0.2:
        return False
    min_width = min(bt.width for bt in BTYPES.values())

    intervals = low_foundation_intervals(before, owner) + [(x0, x1)]
    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + ALIGN_TOL:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    left_gap = max(0.0, x0 - span0)
    right_gap = max(0.0, span1 - x1)
    left_dead = 0.2 < left_gap < min_width - 0.2 and not any(m[1] > span0 + ALIGN_TOL and m[1] <= x0 + ALIGN_TOL for m in merged if m[1] <= x0 + ALIGN_TOL)
    right_dead = 0.2 < right_gap < min_width - 0.2 and not any(m[0] < span1 - ALIGN_TOL and m[0] >= x1 - ALIGN_TOL for m in merged if m[0] >= x1 - ALIGN_TOL)
    return left_dead and right_dead


def inset_over_effective_cantilever(before: Bridge, candidate: BrickInst, side: str, closes_bridge: bool) -> bool:
    if closes_bridge:
        return False
    inset_tol = 1.0
    min_overlap = 1.0
    for lower in before.bricks:
        if side_for_x((lower.x0 + lower.x1) / 2.0) != side:
            continue
        if reach_for_side(lower, side) <= 0.05:
            continue
        if candidate.wz <= lower.wz + ALIGN_TOL:
            continue
        hit = overlap(candidate.x0, candidate.x1, lower.x0, lower.x1)
        if not hit or hit[1] - hit[0] < min_overlap:
            continue
        if side == "L" and candidate.x1 < lower.x1 - inset_tol:
            return True
        if side == "R" and candidate.x0 > lower.x0 + inset_tol:
            return True
    return False


def conservative_rear_fill_score(before: Bridge, candidate: BrickInst, side: str, reach_gain: float) -> float:
    if reach_gain > 0.05:
        return 0.0
    same = side_bricks_on_layer(before, side, candidate.wz)
    if not same:
        return 0.0
    if side == "L":
        front = max(b.x1 for b in same)
        if candidate.x1 > front + ALIGN_TOL:
            return 0.0
        rear_gap = max(0.0, front - candidate.x1)
    else:
        front = min(b.x0 for b in same)
        if candidate.x0 < front - ALIGN_TOL:
            return 0.0
        rear_gap = max(0.0, candidate.x0 - front)
    return min(1.0, rear_gap / max(1.0, LAYOUT.base_w))


def layer_intervals(bridge: Bridge, side: str, z: float) -> List[Tuple[float, float]]:
    return merged_intervals(
        [
            (b.x0, b.x1)
            for b in bridge.bricks
            if side_for_x((b.x0 + b.x1) / 2.0) == side and same_layer(b.wz, z)
        ]
    )


def interval_overlap_width(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for a0, a1 in a:
        for b0, b1 in b:
            total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def side_layer_levels(bridge: Bridge, side: str) -> List[float]:
    levels: List[float] = []
    for b in bridge.bricks:
        if side_for_x((b.x0 + b.x1) / 2.0) != side:
            continue
        z = layer_key(b.wz)
        if not any(abs(z - old) <= ALIGN_TOL for old in levels):
            levels.append(z)
    return sorted(levels)


def layer_coverage_by_next_layer(bridge: Bridge, side: str, lower_z: float, upper_z: float) -> float:
    lower = layer_intervals(bridge, side, lower_z)
    if not lower:
        return 1.0
    upper = layer_intervals(bridge, side, upper_z)
    return min(1.0, interval_overlap_width(lower, upper) / max(1e-6, union_width(lower)))


def conservative_layer_constraint_ok(before: Bridge, after: Bridge, candidate: BrickInst, side: str, closes_bridge: bool) -> bool:
    return True


def conservative_single_layer_ok(before: Bridge, after: Bridge, candidate: BrickInst, side: str, closes_bridge: bool) -> bool:
    if closes_bridge:
        return True
    cand_z = layer_key(candidate.wz)
    for debt_side in ("L", "R"):
        for z in side_layer_levels(before, debt_side):
            if len(side_bricks_on_layer(before, debt_side, z)) == 1:
                return side == debt_side and same_layer(candidate.wz, z)
    after_layer = side_bricks_on_layer(after, side, candidate.wz)
    if len(after_layer) == 1:
        if is_foundation_move(candidate):
            span0, span1 = base_span_for_owner(owner_for_side(side))
            clipped = max(0.0, min(candidate.x1, span1) - max(candidate.x0, span0))
            base_fit = clipped / max(1e-6, candidate.x1 - candidate.x0)
            return candidate.btype.width <= SINGLE_LAYER_MAX_BRICK_WIDTH and base_fit >= 0.86
        return candidate.btype.width <= SINGLE_LAYER_MAX_BRICK_WIDTH and lower_layer_fill_score(before, candidate, side) >= 0.65
    return True


def lower_before_upper_ok(before: Bridge, candidate: BrickInst, side: str, closes_bridge: bool) -> bool:
    if closes_bridge:
        return True
    levels = side_layer_levels(before, side)
    if not levels:
        return True
    return layer_key(candidate.wz) >= max(levels) - ALIGN_TOL


def layer_fillable_gaps(bridge: Bridge, side: str, z: float) -> List[Tuple[float, float]]:
    intervals = layer_intervals(bridge, side, z)
    if not intervals:
        return []
    min_width = min(bt.width for bt in BTYPES.values()) - 0.2
    gaps: List[Tuple[float, float]] = []

    if z <= low_layer_limit() + ALIGN_TOL:
        span0, span1 = base_span_for_owner(owner_for_side(side))
        cursor = span0
        for x0, x1 in intervals:
            gap = (cursor, min(x0, span1))
            if gap[1] - gap[0] >= min_width:
                gaps.append(gap)
            cursor = max(cursor, x1)
        gap = (cursor, span1)
        if gap[1] - gap[0] >= min_width:
            gaps.append(gap)
        return gaps

    # Above the foundation, only internal holes are treated as blocking. Outer
    # expansion is how cantilevers grow; internal gaps are the places a later
    # brick would have to be inserted underneath an already higher structure.
    for (_a0, a1), (b0, _b1) in zip(intervals, intervals[1:]):
        if b0 - a1 >= min_width:
            gaps.append((a1, b0))
    return gaps


def blocks_future_lower_fill(before: Bridge, candidate: BrickInst, side: str, closes_bridge: bool) -> bool:
    if closes_bridge:
        return False
    levels = side_layer_levels(before, side)
    if not levels:
        return False
    top = max(levels)
    if layer_key(candidate.wz) <= top + ALIGN_TOL:
        return False
    return bool(layer_fillable_gaps(before, side, top))


def lower_layer_fill_score(before: Bridge, candidate: BrickInst, side: str) -> float:
    cand_z = layer_key(candidate.wz)
    lower_levels = [z for z in side_layer_levels(before, side) if z < cand_z - ALIGN_TOL]
    if not lower_levels:
        return 0.0
    lower = layer_intervals(before, side, max(lower_levels))
    if not lower:
        return 0.0
    candidate_span = [(candidate.x0, candidate.x1)]
    return min(1.0, interval_overlap_width(candidate_span, lower) / max(1e-6, candidate.x1 - candidate.x0))


def low_height_bonus(bi: BrickInst) -> float:
    return max(0.0, 1.0 - bi.wz / max(1e-6, low_layer_limit() + HEX_RISE))


NEXT_REACH_CACHE: Dict[Tuple, float] = {}


def next_stable_reach_potential(bridge: Bridge, side: str, required_margin: float) -> float:
    cache_key = ("next_reach", bridge_state_key(bridge), side, round(required_margin, 3))
    cached = NEXT_REACH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    best = 0.0
    for bt in BTYPES.values():
        for wx in candidate_x_positions(bridge, bt, side):
            for option in bridge.try_place_options(bt, wx, "robot", max_options=2):
                sim = bridge.clone()
                bi = BrickInst(
                    sim.next_id,
                    option.btype,
                    option.wx,
                    option.wz,
                    "robot",
                    list(option.contacts),
                )
                sim.bricks.append(bi)
                sim.next_id += 1
                sim.analyse()
                if sim.min_margin >= required_margin or sim.is_closing_brick(bi.id):
                    best = max(best, side_reach_gain(bridge, sim, side))
    cache_put(NEXT_REACH_CACHE, cache_key, best)
    return best


def premature_height_penalty(bridge: Bridge, bi: BrickInst) -> float:
    if bridge.gap_coverage > 0.65:
        return 0.0
    height_limit = low_layer_limit()
    return max(0.0, bi.wz - height_limit)


def snap_place_x(wx: float) -> float:
    """Snap placements to the projected hex-lattice pitch."""
    return snap_to(wx, PLACE_GRID)


def brick_bounds_center(bt: BrickType) -> float:
    return (bt.x_min + bt.x_max) / 2


def candidate_x_positions(bridge: Bridge, bt: BrickType, side: str) -> List[float]:
    supports = support_polys(bridge.bricks)
    support_edges = lattice_support_edges(bridge.bricks)
    local_lower_edges = brick_cell_edges(bt, 0.0, 0.0, "lower", 0)
    candidates: set[float] = set()

    for lx0, lz0, lx1, lz1, _ in local_lower_edges:
        ldx = lx1 - lx0
        ldz = lz1 - lz0
        for sx0, sz0, sx1, sz1, _owner in support_edges:
            if abs((sx1 - sx0) - ldx) > CONTACT_TOL:
                continue
            if abs((sz1 - sz0) - ldz) > CONTACT_TOL:
                continue
            candidates.add(round(snap_place_x(sx0 - lx0), 6))

    result: List[float] = []
    for xi in sorted(candidates):
        wx = float(xi)
        center = wx + brick_bounds_center(bt)
        if side_for_x(center) == side:
            result.append(wx)
    return result


def future_potential(bridge: Bridge, strategy: str) -> float:
    if bridge.bridge_closed:
        return 10.0
    margin = 0.0 if bridge.min_margin == math.inf else max(0.0, bridge.min_margin)
    balance = max(0.0, 1.0 - reach_balance(bridge) / max(1.0, GAP_W))
    progress = bridge.gap_coverage
    low_fill = foundation_coverage(bridge)
    if strategy == "conservative":
        return 1.0 * progress + 1.35 * balance + 1.65 * min(1.0, margin) + 2.35 * low_fill
    return 3.75 * progress + 1.0 * balance + 0.70 * min(1.0, margin) + 0.80 * low_fill


def recommend_immediate(bridge: Bridge, strategy: str, limit: int = 12) -> List[Rec]:
    if bridge.bridge_succeeded or bridge.brick_exhausted:
        return []
    base_margin = bridge.min_margin if bridge.joints else 1.0
    required_margin = CONSERVATIVE_MARGIN if strategy == "conservative" else MIN_ACCEPT_MARGIN
    prep_margin = CONSERVATIVE_PREP_MARGIN if strategy == "conservative" else AGGRESSIVE_PREP_MARGIN
    base_remaining = bridge.remaining_gap
    base_balance = reach_balance(bridge)
    base_foundation = foundation_average_coverage(bridge)
    progress_streak = no_progress_streak(bridge)
    force_progress = strategy == "conservative" and progress_streak >= 4
    near_closing = strategy == "conservative" and base_remaining <= CONSERVATIVE_CLOSE_PRIORITY_GAP_W + CONTACT_TOL
    cands: List[Rec] = []
    urgent_reinforce_cands: List[Rec] = []
    secondary_cands: List[Rec] = []
    fallback_cands: List[Rec] = []
    sides = (active_user_side(bridge),)
    if sides[0] is None:
        sides = ("L", "R")

    for side in sides:
        side_foundation_before = foundation_coverage_for_side(bridge, side)
        side_util_before = base_cell_utilization_for_side(bridge, side)
        side_void_before = base_trapped_void_score_for_side(bridge, side)
        needs_foundation = foundation_stage_required_for_side(bridge, strategy, side)
        foundation_ready = not needs_foundation
        for bt in BTYPES.values():
            for wx in candidate_x_positions(bridge, bt, side):
                for option in bridge.try_place_options(bt, wx, "robot"):
                    sim = bridge.clone()
                    bi = BrickInst(
                        sim.next_id,
                        option.btype,
                        option.wx,
                        option.wz,
                        "robot",
                        list(option.contacts),
                    )
                    sim.bricks.append(bi)
                    sim.next_id += 1
                    sim.analyse()
                    margin_after = sim.min_margin
                    dm = margin_after - base_margin
                    closure_gain = max(0.0, base_remaining - sim.remaining_gap)
                    reach_gain = side_reach_gain(bridge, sim, side)
                    balance_gain = base_balance - reach_balance(sim)
                    coverage_after = sim.gap_coverage
                    width = contact_width(bi.contacts)
                    support_norm = normalized_contact_width(bi)
                    margin_norm = normalized_margin(margin_after)
                    foundation = foundation_score(bridge, bi)
                    side_foundation_gain = max(0.0, foundation_coverage_for_side(sim, side) - side_foundation_before)
                    side_util_after = base_cell_utilization_for_side(sim, side)
                    side_util_gain = max(0.0, side_util_after - side_util_before)
                    side_void_after = base_trapped_void_score_for_side(sim, side)
                    void_reduction = max(0.0, side_void_before - side_void_after)
                    trapped_void_penalty = max(0.0, side_void_after - side_void_before)
                    foundation_gain = max(0.0, foundation_average_coverage(sim) - base_foundation)
                    packing = foundation_packing_score(sim)
                    support_efficiency = foundation_support_efficiency(bi)
                    joint_reinforcement = joint_reinforcement_score(bridge, bi)
                    urgent_reinforcement = (
                        urgent_joint_reinforcement_score(bridge, bi, side)
                        if strategy == "conservative" and not near_closing
                        else 0.0
                    )
                    low_bonus = low_height_bonus(bi)
                    height_penalty = premature_height_penalty(bridge, bi)
                    stagnant_height_penalty = (
                        max(0.0, bi.wz - low_layer_limit())
                        if reach_gain < 0.05 and side_foundation_gain < 0.02
                        else 0.0
                    )
                    spans_both = sim.is_closing_brick(bi.id)
                    closes_bridge = strategy_closes_bridge(bridge, sim, bi, strategy)
                    if not support_contact_ok(bi, closes_bridge):
                        continue
                    if not lower_before_upper_ok(bridge, bi, side, closes_bridge):
                        continue
                    if strategy == "conservative" and blocks_future_lower_fill(bridge, bi, side, closes_bridge):
                        continue
                    if strategy == "conservative" and spans_both and not closes_bridge:
                        continue
                    if strategy == "conservative" and creates_unusable_foundation_island(bridge, bi, side):
                        continue
                    if strategy == "conservative" and not conservative_layer_constraint_ok(bridge, sim, bi, side, closes_bridge):
                        continue
                    if not conservative_single_layer_ok(bridge, sim, bi, side, closes_bridge):
                        continue
                    tower_stack = (
                        not closes_bridge
                        and reach_gain < 0.05
                        and side_util_gain < 0.02
                        and side_foundation_gain < 0.02
                        and void_reduction < 0.02
                        and bi.wz > low_layer_limit() + HEX_RISE
                    )
                    if tower_stack:
                        continue
                    if not closes_bridge and not stable_enough_margin(margin_after):
                        continue
                    margin_ok = margin_after >= required_margin
                    intent = "cantilever" if closure_gain >= 0.05 else "reinforce"
                    prep_penalty = 0.60 if closure_gain < 0.05 and base_margin >= prep_margin else 0.0
                    early_overhang_penalty = (
                        3.0 * reach_gain if strategy == "conservative" and side_util_before < 0.60 else 0.0
                    )
                    early_overfill_penalty = (
                        20.0 * max(0.0, side_util_after - 0.78)
                        if strategy == "conservative" and side_util_before < 0.15
                        else 0.0
                    )
                    if strategy == "conservative":
                        rear_fill = conservative_rear_fill_score(bridge, bi, side, reach_gain)
                        lower_fill = lower_layer_fill_score(bridge, bi, side)
                        if side_util_before < 0.60:
                            base_weight = 7.0
                            reach_weight = 0.95
                            no_progress_weight = 0.80
                        elif side_util_before < 0.80:
                            base_weight = 4.5
                            reach_weight = 2.20
                            no_progress_weight = 1.45
                        else:
                            base_weight = 1.8
                            reach_weight = 6.00
                            no_progress_weight = 7.50
                        capped_margin = min(margin_norm, 0.85)
                        mature_no_progress_penalty = (
                            4.0 if side_util_before >= 0.80 and reach_gain < 0.05 and not closes_bridge else 0.0
                        )
                        score = (
                            1.45 * capped_margin
                            + 1.60 * support_norm
                            + 0.95 * balance_gain
                            + 0.75 * min(1.0, coverage_after)
                            + base_weight * side_util_gain
                            + 2.60 * void_reduction
                            + 1.20 * side_foundation_gain
                            + 1.90 * foundation_gain
                            + 1.10 * packing
                            + 0.80 * support_efficiency
                            + 2.10 * joint_reinforcement
                            + 9.00 * urgent_reinforcement
                            + 0.45 * low_bonus
                            + 2.80 * rear_fill
                            + 2.40 * lower_fill
                            + reach_weight * reach_gain
                            + (4.25 if near_closing else 1.15) * closure_gain
                            - 0.80 * max(0.0, -dm)
                            - early_overhang_penalty
                            - early_overfill_penalty
                            - 1.55 * height_penalty
                            - no_progress_weight * stagnant_height_penalty
                            - mature_no_progress_penalty
                            - 2.20 * trapped_void_penalty
                            - (0.25 if not foundation_ready else 0.0) * max(0.0, reach_gain - max(0.0, side_util_gain) * 6.0)
                            - 0.40 * prep_penalty
                        )
                    else:
                        score = (
                            0.55 * margin_after
                            + 0.25 * width
                            + 1.85 * closure_gain
                            + 0.55 * balance_gain
                            + 0.45 * coverage_after
                            + 1.35 * joint_reinforcement
                            + 0.55 * foundation
                            + 2.15 * foundation_gain
                            - 0.15 * max(0.0, -dm)
                            - 0.45 * height_penalty
                            - prep_penalty
                        )
                    if closes_bridge:
                        intent = "close"
                        score += 100.0 + 10.0 * max(0.0, margin_after)
                    if not margin_ok and not closes_bridge:
                        score -= 30.0 + 8.0 * max(0.0, required_margin - margin_after)
                    if intent == "reinforce" and strategy == "conservative":
                        score += 0.25
                    if needs_foundation and is_foundation_move(bi):
                        if strategy == "conservative":
                            rear_fill = conservative_rear_fill_score(bridge, bi, side, reach_gain)
                            lower_fill = lower_layer_fill_score(bridge, bi, side)
                            if side_util_before < 0.60:
                                base_weight = 10.0
                                reach_weight = 0.70
                            elif side_util_before < 0.80:
                                base_weight = 5.5
                                reach_weight = 1.70
                            else:
                                base_weight = 1.5
                                reach_weight = 5.00
                            mature_no_progress_penalty = (
                                4.0 if side_util_before >= 0.80 and reach_gain < 0.05 and not closes_bridge else 0.0
                            )
                            score = (
                                base_weight * side_util_gain
                                + 3.50 * void_reduction
                                + 2.20 * side_foundation_gain
                                + 2.40 * foundation_gain
                                + 2.70 * packing
                                + 2.00 * support_efficiency
                                + 2.30 * joint_reinforcement
                                + 9.50 * urgent_reinforcement
                                + 1.40 * support_norm
                                + 1.00 * min(margin_norm, 0.85)
                                + 0.95 * low_bonus
                                + 3.20 * rear_fill
                                + 1.80 * lower_fill
                                - 0.35 * height_penalty
                                + reach_weight * reach_gain
                                - early_overhang_penalty
                                - early_overfill_penalty
                                - mature_no_progress_penalty
                                - 2.40 * trapped_void_penalty
                            )
                        else:
                            score = (
                                5.0 * foundation_gain
                                + 0.90 * foundation
                                + 1.80 * packing
                                + 1.80 * foundation_open_slot_score(sim)
                                + 1.40 * support_efficiency
                                + 1.20 * joint_reinforcement
                                + 0.45 * min(1.0, margin_after)
                                + 0.08 * width
                                - 0.15 * height_penalty
                            )
                    rec = Rec(side, bt, wx, bi.wz, bi.contacts, score, margin_after, dm, reach_gain, intent)
                    if strategy == "conservative":
                        classified = False
                        if closes_bridge:
                            cands.append(rec)
                            classified = True
                        elif near_closing and reach_gain > 0.05:
                            cands.append(rec)
                            classified = True
                        elif urgent_reinforcement > 0.10:
                            urgent_reinforce_cands.append(rec)
                            classified = True
                        elif force_progress and reach_gain > 0.05:
                            cands.append(rec)
                            classified = True
                        elif force_progress:
                            fallback_cands.append(rec)
                            classified = True
                        elif not margin_ok:
                            fallback_cands.append(rec)
                            classified = True
                        elif side_util_before < 0.65:
                            if side_util_gain > 0.02 and is_foundation_move(bi):
                                cands.append(rec)
                                classified = True
                            elif reach_gain > 0.05:
                                secondary_cands.append(rec)
                                classified = True
                            elif side_util_gain > 0.02 or void_reduction > 0.02:
                                secondary_cands.append(rec)
                                classified = True
                        else:
                            if reach_gain > 0.05:
                                cands.append(rec)
                                classified = True
                            elif side_util_gain > 0.02 or void_reduction > 0.02:
                                secondary_cands.append(rec)
                                classified = True
                            elif (
                                side_foundation_gain > 0.02
                                and bi.wz <= low_layer_limit() + HEX_RISE
                            ):
                                secondary_cands.append(rec)
                                classified = True
                            elif (
                                margin_after > base_margin + 0.10
                                and bi.wz <= low_layer_limit() + HEX_RISE
                            ):
                                fallback_cands.append(rec)
                                classified = True
                        if not classified:
                            fallback_cands.append(rec)
                    elif needs_foundation and not is_foundation_move(bi):
                        fallback_cands.append(rec)
                    else:
                        cands.append(rec)

    if urgent_reinforce_cands:
        active = urgent_reinforce_cands
        active.sort(key=lambda r: (-r.score, -r.margin_after, r.wz, r.side, r.bt.key, r.wx))
    elif cands:
        active = cands
        if strategy == "conservative" and force_progress:
            active.sort(key=lambda r: (-r.reach_gain, -r.margin_after, r.wz, r.side, r.bt.key, r.wx))
        else:
            active.sort(key=lambda r: (-r.score, r.side, r.bt.key, r.wx))
    elif secondary_cands:
        active = secondary_cands
        active.sort(key=lambda r: (-r.score, r.side, r.bt.key, r.wx))
    else:
        active = fallback_cands
        if strategy == "conservative":
            active.sort(key=lambda r: (-r.reach_gain, r.wz, -r.margin_after, r.side, r.bt.key, r.wx))
        else:
            active.sort(key=lambda r: (-r.score, r.side, r.bt.key, r.wx))
    return active[:limit]


def rec_is_placeable(bridge: Bridge, rec: Rec) -> bool:
    placed = bridge.try_place(rec.bt, rec.wx, "robot", rec.wz)
    if not placed:
        return False
    if material_overlaps_existing(placed.btype, placed.wx, placed.wz, bridge.bricks):
        return False
    if contact_width(placed.contacts) < 0.2:
        return False
    sim = bridge.clone()
    bi = BrickInst(
        sim.next_id,
        placed.btype,
        placed.wx,
        placed.wz,
        "robot",
        list(placed.contacts),
    )
    sim.bricks.append(bi)
    sim.next_id += 1
    sim.analyse()
    closes_bridge = sim.is_closing_brick(bi.id)
    if not support_contact_ok(bi, closes_bridge):
        return False
    return stable_enough_margin(sim.min_margin) or closes_bridge


def rec_satisfies_strategy_constraints(bridge: Bridge, rec: Rec, strategy: str) -> bool:
    placed = bridge.try_place(rec.bt, rec.wx, "robot", rec.wz)
    if not placed:
        return False
    sim = bridge.clone()
    bi = BrickInst(sim.next_id, placed.btype, placed.wx, placed.wz, "robot", list(placed.contacts))
    sim.bricks.append(bi)
    sim.next_id += 1
    sim.analyse()
    spans_both = sim.is_closing_brick(bi.id)
    closes_bridge = strategy_closes_bridge(bridge, sim, bi, strategy)
    if not support_contact_ok(bi, closes_bridge):
        return False
    if not lower_before_upper_ok(bridge, bi, rec.side, closes_bridge):
        return False
    if strategy == "conservative" and blocks_future_lower_fill(bridge, bi, rec.side, closes_bridge):
        return False
    if strategy == "conservative" and spans_both and not closes_bridge:
        return False
    if strategy == "conservative" and creates_unusable_foundation_island(bridge, bi, rec.side):
        return False
    if not conservative_single_layer_ok(bridge, sim, bi, rec.side, closes_bridge):
        return False
    if strategy == "aggressive":
        return True
    if strategy == "conservative":
        return conservative_layer_constraint_ok(bridge, sim, bi, rec.side, closes_bridge)
    return True


def emergency_recommend(bridge: Bridge, strategy: str, top_n: int = 4) -> List[Rec]:
    if bridge.bridge_succeeded or bridge.brick_exhausted:
        return []
    sides: List[str] = []
    focus = active_user_side(bridge)
    if focus:
        sides.append(focus)
        sides.append("R" if focus == "L" else "L")
    else:
        sides = ["L", "R"]

    cands: List[Rec] = []
    base_margin = bridge.min_margin if bridge.joints else 1.0
    base_remaining = bridge.remaining_gap
    for side in sides:
        for bt in BTYPES.values():
            for wx in candidate_x_positions(bridge, bt, side):
                for option in bridge.try_place_options(bt, wx, "robot", max_options=8):
                    sim = bridge.clone()
                    bi = BrickInst(
                        sim.next_id,
                        option.btype,
                        option.wx,
                        option.wz,
                        "robot",
                        list(option.contacts),
                    )
                    sim.bricks.append(bi)
                    sim.next_id += 1
                    sim.analyse()
                    margin_after = sim.min_margin
                    reach_gain = side_reach_gain(bridge, sim, side)
                    closure_gain = max(0.0, base_remaining - sim.remaining_gap)
                    spans_both = sim.is_closing_brick(bi.id)
                    closes_bridge = strategy_closes_bridge(bridge, sim, bi, strategy)
                    if not support_contact_ok(bi, closes_bridge):
                        continue
                    if not lower_before_upper_ok(bridge, bi, side, closes_bridge):
                        continue
                    if strategy == "conservative" and blocks_future_lower_fill(bridge, bi, side, closes_bridge):
                        continue
                    if strategy == "conservative" and spans_both and not closes_bridge:
                        continue
                    if strategy == "conservative" and creates_unusable_foundation_island(bridge, bi, side):
                        continue
                    if not conservative_single_layer_ok(bridge, sim, bi, side, closes_bridge):
                        continue
                    if not closes_bridge and not stable_enough_margin(margin_after):
                        continue
                    joint_reinforcement = joint_reinforcement_score(bridge, bi)
                    intent = "close" if closes_bridge else ("cantilever" if reach_gain > 0.05 else "reinforce")
                    score = (
                        20.0
                        + 6.0 * reach_gain
                        + 4.0 * closure_gain
                        + 2.0 * joint_reinforcement
                        + 2.0 * min(1.0, max(-1.0, margin_after))
                        - 0.20 * max(0.0, option.wz - low_layer_limit())
                    )
                    if closes_bridge:
                        score += 100.0
                    cands.append(
                        Rec(side, bt, wx, option.wz, list(option.contacts), score, margin_after, margin_after - base_margin, reach_gain, intent)
                    )
    cands.sort(key=lambda r: (-r.score, -r.margin_after, -r.reach_gain, r.wz, r.side, r.bt.key, r.wx))
    return cands[:top_n]


def guaranteed_recommend(bridge: Bridge, strategy: str, top_n: int = 4) -> List[Rec]:
    if bridge.bridge_succeeded or bridge.brick_exhausted:
        return []
    sides: List[str] = []
    focus = active_user_side(bridge)
    if focus:
        sides.extend([focus, "R" if focus == "L" else "L"])
    else:
        sides.extend(["L", "R"])

    cands: List[Rec] = []
    base_margin = bridge.min_margin if bridge.joints else 1.0
    base_remaining = bridge.remaining_gap
    for side in sides:
        for bt in BTYPES.values():
            for wx in candidate_x_positions(bridge, bt, side):
                for option in bridge.try_place_options(bt, wx, "robot", max_options=12):
                    sim = bridge.clone()
                    bi = BrickInst(sim.next_id, option.btype, option.wx, option.wz, "robot", list(option.contacts))
                    sim.bricks.append(bi)
                    sim.next_id += 1
                    sim.analyse()
                    spans_both = sim.is_closing_brick(bi.id)
                    closes_bridge = strategy_closes_bridge(bridge, sim, bi, strategy)
                    if strategy == "conservative" and spans_both and not closes_bridge:
                        continue
                    if not support_contact_ok(bi, closes_bridge):
                        continue
                    if not closes_bridge and not stable_enough_margin(sim.min_margin):
                        continue
                    reach_gain = side_reach_gain(bridge, sim, side)
                    closure_gain = max(0.0, base_remaining - sim.remaining_gap)
                    joint_reinforcement = joint_reinforcement_score(bridge, bi)
                    intent = "close" if closes_bridge else ("cantilever" if reach_gain > 0.05 else "reinforce")
                    score = (
                        100.0 * (1 if closes_bridge else 0)
                        + 4.0 * closure_gain
                        + 2.0 * reach_gain
                        + 1.5 * joint_reinforcement
                        + 1.2 * normalized_margin(sim.min_margin)
                        + 0.5 * normalized_contact_width(bi)
                        - 0.05 * option.wz
                    )
                    cands.append(
                        Rec(side, bt, wx, option.wz, list(option.contacts), score, sim.min_margin, sim.min_margin - base_margin, reach_gain, intent)
                    )
    cands = [rec for rec in cands if rec_is_placeable(bridge, rec)]
    cands.sort(key=lambda r: (-r.score, -r.margin_after, -r.reach_gain, r.wz, r.side, r.bt.key, r.wx))
    return cands[:top_n]


def recommend(bridge: Bridge, strategy: str, top_n: int = 4) -> List[Rec]:
    cache_key = ("recommend", bridge_state_key(bridge), strategy, top_n)
    cached = RECOMMEND_CACHE.get(cache_key)
    if cached is not None:
        return [copy_rec(rec) for rec in cached]

    cands = recommend_immediate(bridge, strategy, limit=max(8, top_n * 2))
    lookahead_weight = 0.35 if strategy == "conservative" else 0.55
    for rec in cands[:top_n]:
        sim = bridge.clone()
        if sim.place(rec.bt, rec.wx, "robot", rec.wz):
            rec.score += lookahead_weight * future_potential(sim, strategy)
    cands.sort(key=lambda r: (-r.score, r.side, r.bt.key, r.wx))
    result = [
        rec for rec in cands
        if rec_is_placeable(bridge, rec) and rec_satisfies_strategy_constraints(bridge, rec, strategy)
    ][:top_n]
    if strategy != "aggressive":
        if not result:
            result = [
                rec for rec in emergency_recommend(bridge, strategy, max(50, top_n * 20))
                if rec_is_placeable(bridge, rec) and rec_satisfies_strategy_constraints(bridge, rec, strategy)
            ][:top_n]
    if not result:
        result = guaranteed_recommend(bridge, strategy, top_n)
    cache_put(RECOMMEND_CACHE, cache_key, [copy_rec(rec) for rec in result])
    return [copy_rec(rec) for rec in result]


def evaluate_human(before: Bridge, after: Bridge, placed: BrickInst) -> Tuple[str, str]:
    if after.is_closing_brick(placed.id):
        return "nod", "Robot nods: closing brick accepted"
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


def draw_dashed_line(
    sc: pygame.Surface,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    col: Tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    dist = 0.0
    while dist < length:
        end = min(length, dist + dash)
        a = (int(x0 + ux * dist), int(y0 + uy * dist))
        b = (int(x0 + ux * end), int(y0 + uy * end))
        pygame.draw.line(sc, col, a, b, width)
        dist += dash + gap


def draw_dashed_poly(
    sc: pygame.Surface,
    pts: Sequence[Tuple[int, int]],
    col: Tuple[int, int, int],
    width: int = 2,
) -> None:
    for i in range(len(pts) - 1):
        draw_dashed_line(sc, pts[i], pts[i + 1], col, width)


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

        for b in bridge.bricks:
            self.draw_brick(b)

        if recs:
            r = recs[0]
            pulse = 24 + int(24 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 650.0)))
            self.draw_ghost(r.bt, r.wx, r.wz, pulse, outline_alpha=135)

        if hover_wx is not None and sel in BTYPES:
            bt = BTYPES[sel]
            wx = snap_place_x(hover_wx - brick_bounds_center(bt))
            preview = bridge.try_place(bt, wx, "human")
            if preview:
                self.draw_ghost(bt, wx, preview.wz, 22, outline_alpha=125)

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
        if bridge.bridge_succeeded:
            self.banner("Build succeeded: bridge closed", GREEN, y_offset=-56)
        elif bridge.bridge_closed:
            self.banner("Bridge closed but unstable", ORANGE, y_offset=-56)
        elif bridge.brick_exhausted:
            self.banner("Build failed: bricks exhausted", RED, y_offset=-56)

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
        base = BRICK_COLS[brick_color_key(b.btype)]
        fill = lerp(base, margin_color(b.margin), 0.28)
        outline = tuple(max(0, c - 58) for c in base)
        alpha = 224 if b.actor == "human" else 204
        poly(self.sc, pts, fill + (alpha,), outline)
        cx, cz = w2p(b.world_cx, b.world_cz)
        pygame.draw.circle(self.sc, WHITE, (cx, cz), 5)
        pygame.draw.circle(self.sc, COM, (cx, cz), 3)
        label = self.f12.render(b.btype.key.lower() if b.actor == "human" else b.btype.key, True, (18, 20, 24))
        self.sc.blit(label, (cx - label.get_width() // 2, cz - 18))

    def draw_ghost(self, bt: BrickType, wx: float, wz: float, alpha: int, outline_alpha: int = 120) -> None:
        pts = brick_points(bt, wx, wz)
        col = BRICK_COLS[brick_color_key(bt)]
        layer = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.polygon(layer, col + (alpha,), pts)
        self.sc.blit(layer, (0, 0))
        line_layer = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        draw_dashed_poly(line_layer, pts, col + (outline_alpha,), 2)
        self.sc.blit(line_layer, (0, 0))

    def draw_brick_icon(self, bt: BrickType, rect: pygame.Rect, selected: bool) -> None:
        col = BRICK_COLS[brick_color_key(bt)]
        bg = (35, 37, 45) if not selected else (48, 51, 62)
        pygame.draw.rect(self.sc, bg, rect, border_radius=7)
        if selected:
            pygame.draw.rect(self.sc, WHITE, rect, 2, border_radius=7)

        pad_x, pad_y = 10, 7
        inner_w = rect.width - pad_x * 2
        inner_h = rect.height - pad_y * 2
        scale = min(inner_w / max(0.1, bt.width), inner_h / max(0.1, bt.height))
        shape_w = bt.width * scale
        shape_h = bt.height * scale
        origin_x = rect.centerx - shape_w / 2 - bt.x_min * scale
        origin_y = rect.centery + shape_h / 2 + 2
        pts = [
            (int(origin_x + x * scale), int(origin_y - z * scale))
            for x, z in bt.poly
        ]
        pygame.draw.polygon(self.sc, col, pts)
        pygame.draw.polygon(self.sc, tuple(max(0, c - 55) for c in col), pts, 2)
        label = self.f12.render(bt.key, True, WHITE)
        self.sc.blit(label, (rect.left + 6, rect.top + 4))

    def draw_weakest(self, joint: JointResult) -> None:
        b = joint.brick
        y = w2p(0, max(c.z for c in b.contacts))[1]
        x0, x1 = w2p(joint.sx0, 0)[0], w2p(joint.sx1, 0)[0]
        layer = pygame.Surface((max(1, x1 - x0), 7), pygame.SRCALPHA)
        layer.fill((*margin_color(joint.margin), 95))
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
        focus = active_user_side(bridge) or "both"
        row(f"Base preset: {LAYOUT.base_label}-span   focus {focus}   keys 5/9/3", self.f12, PANEL_MUTED)
        sep()

        danger_col = lerp(GREEN, RED, bridge.danger)
        mm = "--" if bridge.min_margin == math.inf else f"{bridge.min_margin:.2f}"
        row("Stability", self.f16, PANEL_TEXT)
        bar(bridge.danger, danger_col)
        row(f"danger {bridge.danger * 100:.0f}%   min margin {mm}", self.f12, danger_col)
        row(f"coverage {bridge.gap_coverage * 100:.0f}%   L {bridge.left_reach:.1f} / R {bridge.right_reach:.1f}", self.f12, PANEL_MUTED)
        if bridge.bridge_succeeded:
            row(f"status success   bricks {len(bridge.bricks)}/{MAX_BRICKS}", self.f12, GREEN)
        elif bridge.brick_exhausted:
            row(f"status failed   bricks {len(bridge.bricks)}/{MAX_BRICKS}", self.f12, RED)
        else:
            closed = "closed" if bridge.bridge_closed else "open"
            row(f"status {closed}   bricks {len(bridge.bricks)}/{MAX_BRICKS}", self.f12, PANEL_MUTED)
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
            row("Searching move", self.f12, ORANGE)
        for i, r in enumerate(recs[:3]):
            col = BLUE if i == 0 else PANEL_TEXT
            row(f"{i + 1}. {r.side}  brick {r.bt.key}  x={r.wx:.1f}  {r.intent}", self.f12, col, 3)
            row(f"   margin {r.margin_after:.2f}   progress +{r.reach_gain:.1f}", self.f12, PANEL_MUTED, 5)
        sep()

        row("Brick library", self.f16, PANEL_TEXT)
        bx = x
        by = y
        item_w = (width - 10) // 2
        item_h = 42
        for k in BRICK_LIBRARY_KEYS:
            bt = BTYPES[k]
            rect = pygame.Rect(bx, by, item_w, item_h)
            self.draw_brick_icon(bt, rect, k == sel)
            bx += item_w + 10
            if bx + item_w > WIN_W - 18:
                bx = x
                by += item_h + 10
        y = by + item_h + 10
        sep()

        for line in (
            "Left click: human move",
            "Space: robot places ghost",
            "S strategy   Z undo   R reset",
            "A-F brick   T flip   Q quit",
        ):
            row(line, self.f12, PANEL_MUTED, 4)


def robot_place_recommendation(bridge: Bridge, strategy: str) -> Tuple[List[Rec], str, str]:
    if bridge.bridge_succeeded:
        log_event("ROBOT", f"skip: already succeeded | {fmt_state(bridge)}")
        return [], "Build already succeeded", "nod"
    if bridge.brick_exhausted:
        log_event("ROBOT", f"skip: bricks exhausted | {fmt_state(bridge)}")
        return [], "Build failed: bricks exhausted", "shake"
    recs = recommend(bridge, strategy)
    log_recommendations(recs, f"robot strategy={strategy}")
    if not recs:
        log_event("ROBOT", f"emergency empty | {fmt_state(bridge)}")
        return guaranteed_recommend(bridge, strategy, 4), "Robot recalculating move", "shake"
    placed: Optional[BrickInst] = None
    r = recs[0]
    for rec in recs:
        r = rec
        placed = bridge.place(r.bt, r.wx, "robot", r.wz)
        if placed:
            if bridge.is_closing_brick(placed.id) or stable_enough_margin(bridge.min_margin):
                break
            log_event(
                "ROBOT",
                f"rejected unstable candidate {fmt_brick(placed)} margin={bridge.min_margin:.2f} | {fmt_state(bridge)}",
            )
            bridge.remove_id(placed.id)
            placed = None
            continue
        log_event("ROBOT", f"failed candidate {r.bt.key} wx={r.wx:.2f} wz={r.wz:.2f} | {fmt_state(bridge)}")
    if not placed:
        for rec in guaranteed_recommend(bridge, strategy, 8):
            r = rec
            placed = bridge.place(r.bt, r.wx, "robot", r.wz)
            if placed:
                break
        if not placed:
            next_recs = guaranteed_recommend(bridge, strategy, 4)
            log_recommendations(next_recs, "after guaranteed retry")
            return next_recs, "Robot recalculating move", "shake"
    log_event("ROBOT", f"placed {fmt_brick(placed)} | {fmt_state(bridge)}")
    next_recs = recommend(bridge, strategy)
    log_recommendations(next_recs, "after robot")
    return next_recs, f"Robot placed {r.bt.key} on {r.side}", "nod"


def set_base_width(span: int, bridge: Bridge) -> None:
    LAYOUT.base_span = span
    bridge.reset()
    log_event("SCENE", f"base reset span={span} | {fmt_state(bridge)}")


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
    log_event("START", f"app started strategy={strategy} | {fmt_state(bridge)}")
    log_recommendations(recs, "initial")

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
                    log_event("INPUT", f"selected brick {selected}")
                elif ev.key == pygame.K_t:
                    selected = toggle_flip_key(selected)
                    message = f"Selected {selected}"
                    reaction = ""
                    log_event("INPUT", f"flipped selection to {selected}")
                elif key in {"5", "9"}:
                    set_base_width(int(key), bridge)
                    message = f"Base reset to {key}-span"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                    log_recommendations(recs, "after base reset")
                elif ev.key == pygame.K_3:
                    set_base_width(13, bridge)
                    message = "Base reset to 13-span"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                    log_recommendations(recs, "after base reset")
                elif ev.key == pygame.K_s:
                    strategy = "aggressive" if strategy == "conservative" else "conservative"
                    recs = recommend(bridge, strategy)
                    message = f"Strategy changed to {strategy}"
                    reaction = ""
                    log_event("INPUT", f"strategy changed to {strategy} | {fmt_state(bridge)}")
                    log_recommendations(recs, "after strategy")
                elif ev.key == pygame.K_SPACE:
                    recs, message, reaction = robot_place_recommendation(bridge, strategy)
                elif ev.key == pygame.K_z:
                    removed = bridge.undo_last()
                    message = f"Undo {removed.btype.key}" if removed else "Nothing to undo"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                    log_event("UNDO", f"removed={fmt_brick(removed)} | {fmt_state(bridge)}")
                    log_recommendations(recs, "after undo")
                elif ev.key == pygame.K_r:
                    bridge.reset()
                    message = "Scene reset"
                    reaction = ""
                    recs = recommend(bridge, strategy)
                    log_event("SCENE", f"reset | {fmt_state(bridge)}")
                    log_recommendations(recs, "after reset")

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and mx < PANEL_X:
                bt = BTYPES[selected]
                wx = snap_place_x(p2w(mx, my)[0] - brick_bounds_center(bt))
                before = bridge.clone()
                placed = bridge.place(bt, wx, "human")
                if not placed:
                    message = "No support below that placement"
                    reaction = "shake"
                    log_event("HUMAN", f"invalid {selected} wx={wx:.2f} mouse=({mx},{my}) | {fmt_state(bridge)}")
                else:
                    log_event("HUMAN", f"placed {fmt_brick(placed)} | before={fmt_state(before)} after={fmt_state(bridge)}")
                    reaction, message = evaluate_human(before, bridge, placed)
                    log_event("EVAL", f"reaction={reaction} message='{message}' closing={bridge.is_closing_brick(placed.id)} | {fmt_state(bridge)}")
                    if reaction == "remove":
                        bridge.remove_id(placed.id)
                        recs = recommend(bridge, strategy)
                        log_event("HUMAN", f"removed rejected brick id={placed.id} | {fmt_state(bridge)}")
                        log_recommendations(recs, "after reject")
                    else:
                        recs, robot_message, robot_reaction = robot_place_recommendation(bridge, strategy)
                        message = f"{message}; {robot_message}"
                        reaction = robot_reaction if robot_reaction == "shake" else reaction

        ui.render(bridge, recs, selected, strategy, message, reaction, hover_wx)
        clock.tick(30)


if __name__ == "__main__":
    main()
