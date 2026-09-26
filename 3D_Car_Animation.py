
import math
import time
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import *
from OpenGL.GLU import *

CONFIG_PATH = "vantage_config.json"

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    window_size: Tuple[int, int] = (1180, 760)
    fov: float = 35.0

    max_hands: int = 1
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5

    orbit_sensitivity: float = 0.25
    gesture_zoom_gain: float = 10.0
    auto_spin_speed: float = 6.0           # ambient camera drift, degrees/sec

    pinch_threshold: float = 0.055
    pinch_color_thresh: float = 0.032
    swipe_threshold: float = 0.22          # normalized wrist travel to count as a swipe
    palm_rotation_gain: float = 1.15

    assembly_dots_dur: float = 1.6
    assembly_wire_dur: float = 2.2
    assembly_pop_dur: float = 0.6


BASE_PRICE = 34500

COLORS = [
    {"name": "Graphite",              "rgb": (0.16, 0.17, 0.19), "premium": 0},
    {"name": "Arctic White",          "rgb": (0.94, 0.95, 0.97), "premium": 0},
    {"name": "Racing Red",            "rgb": (0.70, 0.09, 0.16), "premium": 900},
    {"name": "Deep Cobalt",           "rgb": (0.11, 0.25, 0.56), "premium": 900},
    {"name": "British Racing Green",  "rgb": (0.04, 0.24, 0.16), "premium": 1200},
    {"name": "Amber Bronze",          "rgb": (0.54, 0.35, 0.17), "premium": 1400},
    {"name": "Liquid Chrome",         "rgb": (0.79, 0.80, 0.82), "premium": 2600},
]
WHEELS = [
    {"name": "Standard 18\"", "rgb": (0.60, 0.63, 0.68), "scale": 1.00, "price": 0},
    {"name": "Sport 19\"",    "rgb": (0.08, 0.09, 0.11), "scale": 1.06, "price": 1100},
    {"name": "Gold Forged",   "rgb": (0.85, 0.66, 0.26), "scale": 1.06, "price": 2400},
    {"name": "Gloss Chrome",  "rgb": (0.91, 0.92, 0.95), "scale": 1.08, "price": 3200},
]
LIGHTS = [
    {"name": "Halogen",     "rgb": (1.00, 0.96, 0.88), "price": 0},
    {"name": "Matrix LED",  "rgb": (0.75, 0.91, 1.00), "price": 1800},
]
SPOILERS = [
    {"name": "None",       "price": 0},
    {"name": "Sport lip",  "price": 650},
    {"name": "GT wing",    "price": 1900},
]
INTERIORS = [
    {"name": "Charcoal",     "rgb": (0.11, 0.11, 0.13), "premium": 0},
    {"name": "Tan Leather",  "rgb": (0.66, 0.51, 0.35), "premium": 1100},
    {"name": "Crimson",      "rgb": (0.43, 0.08, 0.13), "premium": 1600},
]

# Car body side profile (x = length, y = height), closed loop, bottom is flat.
BODY_PROFILE = [
    (-2.05, 0.22), (-2.05, 0.36), (-1.60, 0.56), (-0.85, 0.78),
    (-0.15, 1.24), (0.55, 1.28), (0.85, 1.20), (1.30, 0.72),
    (1.85, 0.50), (2.05, 0.30), (2.05, 0.22),
]
WHEEL_CENTERS = [(-1.28, 0.36, 0.85), (-1.28, 0.36, -0.85),
                 (1.32, 0.36, 0.85), (1.32, 0.36, -0.85)]
AVATAR_POS = (-3.3, 0.0, 1.6)

# ============================================================================
# CAR STATE (customization + pricing + save/load)
# ============================================================================

class CarState:
    def __init__(self):
        self.color = 0
        self.wheel = 0
        self.light = 0
        self.spoiler = 0
        self.interior = 0

    def total_price(self) -> int:
        return (BASE_PRICE + COLORS[self.color]["premium"] + WHEELS[self.wheel]["price"]
                + LIGHTS[self.light]["price"] + SPOILERS[self.spoiler]["price"]
                + INTERIORS[self.interior]["premium"])

    def cycle(self, field: str, step: int = 1):
        lengths = {"color": len(COLORS), "wheel": len(WHEELS), "light": len(LIGHTS),
                   "spoiler": len(SPOILERS), "interior": len(INTERIORS)}
        n = lengths[field]
        setattr(self, field, (getattr(self, field) + step) % n)

    def to_dict(self) -> dict:
        return {"color": self.color, "wheel": self.wheel, "light": self.light,
                "spoiler": self.spoiler, "interior": self.interior}

    def load(self, path: str = CONFIG_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for k in ("color", "wheel", "light", "spoiler", "interior"):
                if k in data:
                    setattr(self, k, data[k])
            return True
        except Exception:
            return False

    def save(self, path: str = CONFIG_PATH):
        data = self.to_dict()
        data["price"] = self.total_price()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================================
# GEOMETRY HELPERS (raw OpenGL immediate mode)
# ============================================================================

def _normal(v0, v1, v2):
    ax, ay, az = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    bx, by, bz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
    length = math.sqrt(nx*nx+ny*ny+nz*nz) or 1.0
    return (nx/length, ny/length, nz/length)


def _quad(v0, v1, v2, v3):
    n = _normal(v0, v1, v2)
    glNormal3fv(n)
    for v in (v0, v1, v2, v3):
        glVertex3fv(v)


def set_material(rgb, emissive=False, shininess=40.0):
    glColor3f(*rgb)
    if emissive:
        glMaterialfv(GL_FRONT, GL_EMISSION, (rgb[0]*0.9, rgb[1]*0.9, rgb[2]*0.9, 1.0))
    else:
        glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
    glMaterialf(GL_FRONT, GL_SHININESS, shininess)


def draw_extruded_profile(profile: List[Tuple[float, float]], half_width: float):
    """Extrudes a closed 2D silhouette sideways along Z to make the car shell."""
    glBegin(GL_QUADS)
    n = len(profile)
    for i in range(n):
        x0, y0 = profile[i]
        x1, y1 = profile[(i+1) % n]
        _quad((x0, y0, -half_width), (x1, y1, -half_width),
              (x1, y1, half_width), (x0, y0, half_width))
    glEnd()
    glBegin(GL_POLYGON)
    glNormal3f(0, 0, -1)
    for x, y in profile:
        glVertex3f(x, y, -half_width)
    glEnd()
    glBegin(GL_POLYGON)
    glNormal3f(0, 0, 1)
    for x, y in reversed(profile):
        glVertex3f(x, y, half_width)
    glEnd()


def draw_box(cx, cy, cz, sx, sy, sz):
    x0, x1 = cx-sx/2, cx+sx/2
    y0, y1 = cy-sy/2, cy+sy/2
    z0, z1 = cz-sz/2, cz+sz/2
    glBegin(GL_QUADS)
    _quad((x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1))
    _quad((x1,y0,z0), (x0,y0,z0), (x0,y1,z0), (x1,y1,z0))
    _quad((x0,y0,z0), (x0,y0,z1), (x0,y1,z1), (x0,y1,z0))
    _quad((x1,y0,z1), (x1,y0,z0), (x1,y1,z0), (x1,y1,z1))
    _quad((x0,y1,z1), (x1,y1,z1), (x1,y1,z0), (x0,y1,z0))
    _quad((x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1))
    glEnd()


def draw_cylinder(radius: float, half_len: float, segments: int = 18):
    """Cylinder with its flat caps facing along Z (axle direction for wheels)."""
    pts = [(radius*math.cos(2*math.pi*i/segments), radius*math.sin(2*math.pi*i/segments))
           for i in range(segments)]
    glBegin(GL_QUADS)
    for i in range(segments):
        x0, y0 = pts[i]
        x1, y1 = pts[(i+1) % segments]
        _quad((x0,y0,-half_len), (x1,y1,-half_len), (x1,y1,half_len), (x0,y0,half_len))
    glEnd()
    for z, normal, order in ((-half_len, (0,0,-1), pts), (half_len, (0,0,1), list(reversed(pts)))):
        glBegin(GL_POLYGON)
        glNormal3fv(normal)
        for x, y in order:
            glVertex3f(x, y, z)
        glEnd()


def draw_cylinder_y(radius: float, half_len: float, segments: int = 12):
    """Same cylinder, reoriented so its axis points along Y (for limbs)."""
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    draw_cylinder(radius, half_len, segments)
    glPopMatrix()


def ease_out_back(x: float) -> float:
    c1, c3 = 1.70158, 2.70158
    x = max(0.0, min(1.0, x))
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


# ============================================================================
# WIREFRAME / POINT-CLOUD DATA (for the progressive build-up animation)
# ============================================================================

def build_wire_geometry():
    hw = 0.83
    n = len(BODY_PROFILE)
    points = [(x, y, -hw) for x, y in BODY_PROFILE] + [(x, y, hw) for x, y in BODY_PROFILE]
    wseg = 12
    for (cx, cy, cz) in WHEEL_CENTERS:
        for i in range(wseg):
            a = 2 * math.pi * i / wseg
            points.append((cx, cy + 0.36*math.cos(a), cz + 0.36*math.sin(a)))

    edges = []
    for side in (0, n):
        for i in range(n):
            edges.append((side+i, side+(i+1) % n))
    for i in range(n):
        edges.append((i, n+i))
    base = 2*n
    for w in range(4):
        for i in range(wseg):
            edges.append((base+w*wseg+i, base+w*wseg+(i+1) % wseg))
    return points, edges


WIRE_POINTS, WIRE_EDGES = build_wire_geometry()


def draw_point_cloud(reveal_count: int):
    glDisable(GL_LIGHTING)
    glColor3f(0.95, 0.62, 0.22)
    glPointSize(5)
    glBegin(GL_POINTS)
    for p in WIRE_POINTS[:max(0, reveal_count)]:
        glVertex3fv(p)
    glEnd()
    glEnable(GL_LIGHTING)


def draw_wireframe(edge_reveal: int):
    glDisable(GL_LIGHTING)
    glColor3f(0.30, 0.82, 1.0)
    glLineWidth(2)
    glBegin(GL_LINES)
    for a, b in WIRE_EDGES[:max(0, edge_reveal)]:
        glVertex3fv(WIRE_POINTS[a])
        glVertex3fv(WIRE_POINTS[b])
    glEnd()
    glColor3f(0.95, 0.62, 0.22)
    glPointSize(4)
    glBegin(GL_POINTS)
    for p in WIRE_POINTS:
        glVertex3fv(p)
    glEnd()
    glEnable(GL_LIGHTING)


# ============================================================================
# CAR + AVATAR RENDERER
# ============================================================================

class CarRenderer:
    def draw_floor(self):
        glDisable(GL_LIGHTING)

        # ---- ground (grass/dirt disc out to the horizon) ----
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(0.10, 0.13, 0.10)
        glVertex3f(0, -0.001, 0)
        glColor3f(0.035, 0.045, 0.04)
        segs = 48
        for i in range(segs + 1):
            a = 2 * math.pi * i / segs
            glVertex3f(14 * math.cos(a), -0.001, 14 * math.sin(a))
        glEnd()

        # ---- road strip running under/around the car ----
        road_len, road_hw = 16.0, 2.6
        curb_w = 0.18
        glBegin(GL_QUADS)
        glColor3f(0.085, 0.086, 0.09)
        glVertex3f(-road_len, 0.0, -road_hw)
        glVertex3f(road_len, 0.0, -road_hw)
        glVertex3f(road_len, 0.0, road_hw)
        glVertex3f(-road_len, 0.0, road_hw)
        glEnd()

        # subtle asphalt center-to-edge shading band (fake ambient occlusion)
        glBegin(GL_QUADS)
        glColor3f(0.05, 0.05, 0.055)
        for side in (-1, 1):
            z0, z1 = side * road_hw, side * (road_hw - 0.5)
            glVertex3f(-road_len, 0.0005, z0)
            glVertex3f(road_len, 0.0005, z0)
            glVertex3f(road_len, 0.0005, z1)
            glVertex3f(-road_len, 0.0005, z1)
        glEnd()

        # concrete curbs on both sides
        glBegin(GL_QUADS)
        glColor3f(0.55, 0.55, 0.53)
        for side in (-1, 1):
            z0 = side * road_hw
            z1 = side * (road_hw + curb_w)
            glVertex3f(-road_len, 0.001, z0)
            glVertex3f(road_len, 0.001, z0)
            glVertex3f(road_len, 0.06, z1)
            glVertex3f(-road_len, 0.06, z1)
        glEnd()

        # dashed center lane line
        glColor3f(0.85, 0.76, 0.30)
        dash_len, gap, half_stripe = 0.7, 0.55, 0.045
        x = -road_len
        glBegin(GL_QUADS)
        while x < road_len:
            x_end = min(x + dash_len, road_len)
            glVertex3f(x, 0.002, -half_stripe)
            glVertex3f(x_end, 0.002, -half_stripe)
            glVertex3f(x_end, 0.002, half_stripe)
            glVertex3f(x, 0.002, half_stripe)
            x += dash_len + gap
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_roadside_props(self):
        """A couple of lamp posts for scale/atmosphere. Lit, so keep GL_LIGHTING on."""
        for side in (-1, 1):
            for x in (-4.5, 4.5):
                glPushMatrix()
                glTranslatef(x, 0.0, side * 3.6)
                set_material((0.10, 0.10, 0.11), shininess=30.0)
                glPushMatrix()
                glTranslatef(0, 1.4, 0)
                draw_cylinder_y(0.045, 1.4, 8)
                glPopMatrix()
                set_material((1.0, 0.95, 0.75), emissive=True)
                draw_box(0, 2.85, 0, 0.16, 0.14, 0.16)
                glPopMatrix()

    def draw_solid_car(self, state: CarState, wheel_angle_deg: float):
        color = COLORS[state.color]
        wheel = WHEELS[state.wheel]
        light = LIGHTS[state.light]
        interior = INTERIORS[state.interior]

        set_material(color["rgb"], shininess=90.0)
        draw_extruded_profile(BODY_PROFILE, 0.83)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        set_material((0.06, 0.07, 0.09), shininess=100.0)
        glColor4f(0.06, 0.07, 0.09, 0.55)
        draw_box(-0.05, 1.0, 0, 1.15, 0.42, 1.42)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        set_material(interior["rgb"])
        draw_box(-0.05, 0.78, 0, 1.0, 0.16, 1.30)

        for (x, y, z) in WHEEL_CENTERS:
            glPushMatrix()
            glTranslatef(x, y, z)
            glScalef(wheel["scale"], wheel["scale"], wheel["scale"])
            glRotatef(wheel_angle_deg, 0, 0, 1)
            set_material((0.05, 0.05, 0.06))
            draw_cylinder(0.36, 0.12, 20)
            set_material(wheel["rgb"], shininess=110.0)
            draw_cylinder(0.20, 0.13, 14)
            for i in range(5):
                glPushMatrix()
                glRotatef(i * 72, 0, 0, 1)
                draw_box(0.13, 0, 0, 0.26, 0.04, 0.04)
                glPopMatrix()
            glPopMatrix()

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        for z in (0.62, -0.62):
            hx, tx = -2.03, 2.03
            # headlight housing, drawn unlit so it reads as a bright light, not a shaded panel
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(light["rgb"][0], light["rgb"][1], light["rgb"][2], 1.0)
            draw_box(hx, 0.5, z, 0.06, 0.12, 0.32)
            glColor4f(1.0, 0.13, 0.15, 1.0)
            draw_box(tx, 0.55, z, 0.06, 0.14, 0.34)

            # additive glow halo around each light so it visibly looks "on"
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glColor4f(light["rgb"][0], light["rgb"][1], light["rgb"][2], 0.35)
            draw_box(hx - 0.05, 0.5, z, 0.20, 0.30, 0.44)
            glColor4f(1.0, 0.15, 0.17, 0.35)
            draw_box(tx + 0.05, 0.55, z, 0.20, 0.32, 0.46)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

        set_material((0.08, 0.09, 0.11), shininess=60.0)
        if state.spoiler == 1:
            draw_box(1.9, 1.15, 0, 0.5, 0.06, 1.5)
        elif state.spoiler == 2:
            for z in (0.55, -0.55):
                draw_box(1.85, 1.28, z, 0.06, 0.32, 0.06)
            draw_box(1.95, 1.44, 0, 0.42, 0.05, 1.6)

    def draw_avatar(self, idle_t: float, paused: bool):
        glPushMatrix()
        glTranslatef(*AVATAR_POS)
        sway = 0.0 if paused else math.sin(idle_t * 1.6) * 4.0

        set_material((0.15, 0.16, 0.20))
        for xo in (-0.13, 0.13):
            glPushMatrix()
            glTranslatef(xo, 0.42, 0)
            draw_cylinder_y(0.09, 0.42, 10)
            glPopMatrix()

        set_material((0.55, 0.14, 0.17))
        draw_box(0, 1.02, 0, 0.5, 0.62, 0.26)

        for xo, sign in ((-0.34, 1), (0.34, -1)):
            glPushMatrix()
            glTranslatef(xo, 1.18, 0)
            glRotatef(sway * sign * 1.4, 1, 0, 0)
            glTranslatef(0, -0.32, 0)
            set_material((0.55, 0.14, 0.17))
            draw_cylinder_y(0.075, 0.32, 10)
            glPopMatrix()

        set_material((0.86, 0.70, 0.58), shininess=20.0)
        glPushMatrix()
        glTranslatef(0, 1.5, 0)
        quad = gluNewQuadric()
        try:
            gluQuadricNormals(quad, GLU_SMOOTH)
        except Exception:
            pass
        gluSphere(quad, 0.19, 16, 12)
        gluDeleteQuadric(quad)
        glPopMatrix()
        glPopMatrix()


# ============================================================================
# HAND TRACKING + GESTURE RECOGNITION
# ============================================================================

class HandTracker:
    def __init__(self, config: Config):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self.connections = list(self.mp_hands.HAND_CONNECTIONS)

    def detect(self, frame: np.ndarray) -> Optional[List[List[Tuple[float, float]]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        return [[(lm.x, lm.y) for lm in hand.landmark] for hand in results.multi_hand_landmarks]

    def close(self):
        self.hands.close()


def _dist(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def is_open_palm(lm) -> bool:
    wrist = lm[0]
    tips, mids = (4, 8, 12, 16, 20), (2, 6, 10, 14, 18)
    extended = sum(1 for t, m in zip(tips, mids) if _dist(lm[t], wrist) > _dist(lm[m], wrist) * 1.25)
    return extended >= 5


def is_two_finger(lm) -> bool:
    wrist = lm[0]
    idx_ext = _dist(lm[8], wrist) > _dist(lm[6], wrist) * 1.2
    mid_ext = _dist(lm[12], wrist) > _dist(lm[10], wrist) * 1.2
    ring_curl = _dist(lm[16], wrist) < _dist(lm[14], wrist) * 1.1
    pinky_curl = _dist(lm[20], wrist) < _dist(lm[18], wrist) * 1.1
    return idx_ext and mid_ext and ring_curl and pinky_curl


def is_pinch(lm, threshold: float) -> bool:
    return _dist(lm[4], lm[8]) < threshold


def count_extended_fingers(lm) -> int:
    """Counts how many of the 5 fingers are extended (0-5), using the same
    tip-vs-knuckle distance trick as is_open_palm, but returning a count
    instead of a yes/no."""
    wrist = lm[0]
    tips, mids = (4, 8, 12, 16, 20), (2, 6, 10, 14, 18)
    return sum(1 for t, m in zip(tips, mids) if _dist(lm[t], wrist) > _dist(lm[m], wrist) * 1.2)


def is_fist(lm) -> bool:
    wrist = lm[0]
    tips, mids = (8, 12, 16, 20), (6, 10, 14, 18)
    curled = sum(1 for t, m in zip(tips, mids) if _dist(lm[t], wrist) < _dist(lm[m], wrist) * 0.95)
    return curled >= 4


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class VantageApp:
    def __init__(self):
        self.config = Config()
        self.state = CarState()
        if self.state.load():
            print(f"[Vantage] Loaded saved configuration from {CONFIG_PATH}")

        pygame.init()
        pygame.display.set_mode(self.config.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Vantage — Car Customizer")
        self.font = pygame.font.SysFont("Arial", 16)

        w, h = self.config.window_size
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(self.config.fov, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 8.0, 4.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.28, 0.30, 0.34, 1.0))
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (-6.0, 3.0, -4.0, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.15, 0.22, 0.28, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.055, 0.06, 0.078, 1.0)

        self.renderer = CarRenderer()

        # scene camera (mouse / scroll / up-down arrows)
        self.cam_azimuth = 28.0
        self.cam_polar = 60.0
        self.cam_distance = 9.5
        self.dragging = False
        self.last_mouse = (0, 0)

        # the car's own turntable rotation (gestures / left-right arrows)
        self.car_rotation = 0.0
        self.car_rotation_target = 0.0

        self.wheel_angle = 0.0
        self.paused = False
        self.scene_clock = 0.0
        self._dt = 1.0 / 60.0

        self.assembly_time = 0.0
        self.assembly_total = (self.config.assembly_dots_dur + self.config.assembly_wire_dur
                                + self.config.assembly_pop_dur)

        self.hand_tracker: Optional[HandTracker] = None
        self.cap = None
        self.gestures_enabled = False
        self._init_camera_source()

        self._swipe_hist = deque()
        self._swipe_cooldown_until = 0.0
        self._pinch_active = False
        self._pinch_base_y = None
        self._pinch_cooldown_until = 0.0
        self._last_two_finger_y = None
        self._rot_prev_angle = None
        self._fist_was_down = False
        self.save_flash_until = 0.0

        self._print_help()

    # ---- setup helpers -----------------------------------------------
    def _init_camera_source(self):
        print("[Vantage] Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[Vantage] No webcam found — hand gestures unavailable, mouse/keyboard still work.")
            self.cap = None
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap = cap
        self.hand_tracker = HandTracker(self.config)
        print("[Vantage] Webcam ready. Press 'g' to enable hand gestures.")

    def _print_help(self):
        print("\n[Vantage] Controls:")
        print("  Mouse drag                orbit camera   Scroll wheel        zoom camera")
        print("  Left / Right arrows       rotate car     Up / Down arrows    tilt camera")
        print("  1-7 / w / l / p / i       paint / wheels / lights / spoiler / interior")
        print("  r                         replay build-up animation")
        print("  Space                     skip straight to the finished car")
        print("  g                         toggle hand gestures")
        print("  s                         save configuration")
        print("  Esc or q                  quit\n")
        print("[Vantage] Gestures (after 'g'): open-hand swipe -> step-rotate car, "
              "twist wrist -> smooth-rotate car,")
        print("          pinch + move up/down -> zoom, hold up 1-4 fingers -> jump to that paint color, "
              "fist -> pause/resume.\n")

    # ---- gesture processing -------------------------------------------
    def _check_swipe(self, x: float, now: float) -> Optional[str]:
        self._swipe_hist.append((now, x))
        while self._swipe_hist and now - self._swipe_hist[0][0] > 0.4:
            self._swipe_hist.popleft()
        if now < self._swipe_cooldown_until or len(self._swipe_hist) < 3:
            return None
        dx = self._swipe_hist[-1][1] - self._swipe_hist[0][1]
        if abs(dx) > self.config.swipe_threshold:
            self._swipe_cooldown_until = now + 0.6
            self._swipe_hist.clear()
            return "right" if dx > 0 else "left"
        return None

    def _reset_gesture_trackers(self):
        self._swipe_hist.clear()
        self._rot_prev_angle = None
        self._pinch_active = False
        self._last_two_finger_y = None

    def _process_gestures(self):
        if not (self.gestures_enabled and self.cap is not None):
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        hands = self.hand_tracker.detect(frame)
        label = "show a hand"
        now = time.time()

        if hands:
            lm = hands[0]
            if is_fist(lm):
                if not self._fist_was_down:
                    self.paused = not self.paused
                    self._fist_was_down = True
                label = f"fist -> {'paused' if self.paused else 'resumed'}"
                self._reset_gesture_trackers()
            elif is_open_palm(lm):
                # checked BEFORE finger-counting so a full open hand always
                # means swipe/rotate, never a color change
                self._fist_was_down = False
                self._pinch_active = False
                self._last_two_finger_y = None
                wrist, mcp = lm[0], lm[9]
                swipe_dir = self._check_swipe(wrist[0], now)
                if swipe_dir == "right":
                    self.car_rotation_target = self.car_rotation + 42.0
                    self._rot_prev_angle = None
                    label = "swipe -> rotate right"
                elif swipe_dir == "left":
                    self.car_rotation_target = self.car_rotation - 42.0
                    self._rot_prev_angle = None
                    label = "swipe -> rotate left"
                else:
                    angle = math.degrees(math.atan2(mcp[1]-wrist[1], mcp[0]-wrist[0]))
                    if self._rot_prev_angle is not None:
                        delta = angle - self._rot_prev_angle
                        while delta > 180: delta -= 360
                        while delta < -180: delta += 360
                        if abs(delta) < 40:
                            self.car_rotation += delta * self.config.palm_rotation_gain
                            self.car_rotation_target = self.car_rotation
                    self._rot_prev_angle = angle
                    label = "palm rotation"
            elif is_pinch(lm, self.config.pinch_threshold):
                # dedicated zoom gesture: pinch, then move hand up/down
                self._fist_was_down = False
                self._swipe_hist.clear()
                self._rot_prev_angle = None
                y = (lm[4][1] + lm[8][1]) / 2.0
                if self._pinch_active and self._last_two_finger_y is not None:
                    self.cam_distance += (y - self._last_two_finger_y) * self.config.gesture_zoom_gain
                    self.cam_distance = max(5.0, min(15.0, self.cam_distance))
                self._pinch_active = True
                self._last_two_finger_y = y
                label = f"pinch -> zoom ({self.cam_distance:.1f})"
            elif 1 <= count_extended_fingers(lm) <= 4:
                # 5 fingers is reserved for open-palm swipe/rotate above, so
                # only 1-4 fingers select a color, avoiding the overlap
                self._fist_was_down = False
                self._pinch_active = False
                self._last_two_finger_y = None
                n = count_extended_fingers(lm)
                idx = n - 1  # 1 finger -> color 0, ..., 4 fingers -> color 3
                if idx != self.state.color:
                    self.state.color = idx
                    label = f"{n} finger(s) -> {COLORS[self.state.color]['name']}"
                else:
                    label = f"{n} finger(s) held"
                self._swipe_hist.clear()
                self._rot_prev_angle = None
            else:
                self._fist_was_down = False
                self._reset_gesture_trackers()
                label = "tracking..."

            self._draw_hand_overlay(frame, lm)
        else:
            self._fist_was_down = False
            self._reset_gesture_trackers()

        cv2.putText(frame, f"Gesture: {label}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "q: close this window", (10, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        preview = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Vantage — Hand Gesture Control", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.gestures_enabled = False
            cv2.destroyWindow("Vantage — Hand Gesture Control")

    def _draw_hand_overlay(self, frame, lm):
        h, w = frame.shape[:2]
        pulse = (math.sin(time.time() * 4.0) + 1.0) / 2.0
        glow = (int(80 + 120*pulse), int(220 - 40*pulse), 255)
        for a, b in self.hand_tracker.connections:
            pa = (int(lm[a][0]*w), int(lm[a][1]*h))
            pb = (int(lm[b][0]*w), int(lm[b][1]*h))
            cv2.line(frame, pa, pb, glow, 2, cv2.LINE_AA)
        for x, y in lm:
            cv2.circle(frame, (int(x*w), int(y*h)), 3, (60, 140, 255), -1, cv2.LINE_AA)

    def _close_gesture_window_if_open(self):
        try:
            if cv2.getWindowProperty("Vantage — Hand Gesture Control", 0) >= 0:
                cv2.destroyWindow("Vantage — Hand Gesture Control")
        except cv2.error:
            pass

    # ---- pygame input ----------------------------------------------------
    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key in (K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_g:
                    self.gestures_enabled = not self.gestures_enabled
                    if not self.gestures_enabled:
                        self._close_gesture_window_if_open()
                    print(f"[Vantage] Hand gestures {'ON' if self.gestures_enabled else 'OFF'}")
                elif event.key == pygame.K_s:
                    self.state.save()
                    self.save_flash_until = time.time() + 1.5
                    print(f"[Vantage] Configuration saved to {CONFIG_PATH}")
                elif event.key == pygame.K_r:
                    self.assembly_time = 0.0
                elif event.key == pygame.K_SPACE:
                    self.assembly_time = self.assembly_total
                elif pygame.K_1 <= event.key <= pygame.K_7:
                    idx = event.key - pygame.K_1
                    if idx < len(COLORS):
                        self.state.color = idx
                elif event.key == pygame.K_w:
                    self.state.cycle("wheel", 1)
                elif event.key == pygame.K_l:
                    self.state.cycle("light", 1)
                elif event.key == pygame.K_p:
                    self.state.cycle("spoiler", 1)
                elif event.key == pygame.K_i:
                    self.state.cycle("interior", 1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse = event.pos
                elif event.button == 4:
                    self.cam_distance = max(5.0, self.cam_distance - 0.4)
                elif event.button == 5:
                    self.cam_distance = min(15.0, self.cam_distance + 0.4)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                dx = event.pos[0] - self.last_mouse[0]
                dy = event.pos[1] - self.last_mouse[1]
                self.last_mouse = event.pos
                self.cam_azimuth += dx * self.config.orbit_sensitivity
                self.cam_polar = max(35.0, min(88.0, self.cam_polar + dy * self.config.orbit_sensitivity))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.car_rotation -= 70 * self._dt
            self.car_rotation_target = self.car_rotation
        if keys[pygame.K_RIGHT]:
            self.car_rotation += 70 * self._dt
            self.car_rotation_target = self.car_rotation
        if keys[pygame.K_UP]:
            self.cam_polar = max(35.0, self.cam_polar - 40 * self._dt)
        if keys[pygame.K_DOWN]:
            self.cam_polar = min(88.0, self.cam_polar + 40 * self._dt)
        return True

    # ---- camera / hud -----------------------------------------------------
    def _update_camera(self):
        az = math.radians(self.cam_azimuth)
        pol = math.radians(self.cam_polar)
        ex = self.cam_distance * math.sin(pol) * math.sin(az)
        ey = self.cam_distance * math.cos(pol)
        ez = self.cam_distance * math.sin(pol) * math.cos(az)
        glLoadIdentity()
        gluLookAt(ex, ey + 0.5, ez, -1.2, 0.7, 0.8, 0, 1, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 8.0, 4.0, 0.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (-6.0, 3.0, -4.0, 0.0))

    def _draw_hud(self):
        w, h = self.config.window_size
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        lines = [
            f"Vantage  —  ${self.state.total_price():,}",
            f"Paint: {COLORS[self.state.color]['name']}   Wheels: {WHEELS[self.state.wheel]['name']}",
            f"Lights: {LIGHTS[self.state.light]['name']}   Spoiler: {SPOILERS[self.state.spoiler]['name']}"
            f"   Interior: {INTERIORS[self.state.interior]['name']}",
            f"Gestures: {'ON' if self.gestures_enabled else 'OFF'} (g)   "
            f"{'PAUSED' if self.paused else 'running'} (fist)   Save (s)   Quit (Esc)",
        ]
        if time.time() < self.save_flash_until:
            lines.append("Configuration saved!")

        # dark translucent backing panel so text has contrast against any background
        panel_h = 14 + 22 * len(lines)
        glColor4f(0.03, 0.035, 0.05, 0.62)
        glBegin(GL_QUADS)
        glVertex2f(8, h - panel_h)
        glVertex2f(8 + 460, h - panel_h)
        glVertex2f(8 + 460, h)
        glVertex2f(8, h)
        glEnd()

        y = h - 26
        for line in lines:
            surf = self.font.render(line, True, (235, 238, 244))
            data = pygame.image.tostring(surf, "RGBA", True)
            glWindowPos2d(16, y)
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)
            y -= 22

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ---- assembly + scene drawing ------------------------------------
    def _draw_car_stage(self):
        cfg = self.config
        t = self.assembly_time
        d1, d2, d3 = cfg.assembly_dots_dur, cfg.assembly_wire_dur, cfg.assembly_pop_dur

        glPushMatrix()
        glRotatef(self.car_rotation, 0, 1, 0)

        if t < d1:
            frac = t / d1
            draw_point_cloud(int(len(WIRE_POINTS) * frac) + 1)
        elif t < d1 + d2:
            frac = (t - d1) / d2
            draw_wireframe(int(len(WIRE_EDGES) * frac))
        else:
            x = (t - d1 - d2) / d3 if d3 > 0 else 1.0
            scale = 0.75 + 0.25 * ease_out_back(x) if x < 1.0 else 1.0
            scale = max(0.7, min(1.12, scale))
            glPushMatrix()
            glScalef(scale, scale, scale)
            self.renderer.draw_solid_car(self.state, self.wheel_angle)
            glPopMatrix()
        glPopMatrix()

    # ---- main loop ----------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            self._dt = clock.tick(60) / 1000.0
            running = self._handle_events()
            self._process_gestures()

            if self.car_rotation_target is not None:
                self.car_rotation += (self.car_rotation_target - self.car_rotation) * min(1.0, self._dt * 6)

            if not self.paused:
                self.scene_clock += self._dt
                self.assembly_time = min(self.assembly_total, self.assembly_time + self._dt)
                self.wheel_angle = (self.wheel_angle + 140 * self._dt) % 360
                if not self.dragging:
                    self.cam_azimuth += self.config.auto_spin_speed * self._dt * 0.15

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._update_camera()
            self.renderer.draw_floor()
            self.renderer.draw_roadside_props()
            self.renderer.draw_avatar(self.scene_clock, self.paused)
            self._draw_car_stage()
            self._draw_hud()
            pygame.display.flip()

        self._shutdown()

    def _shutdown(self):
        if self.cap is not None:
            self.cap.release()
        if self.hand_tracker is not None:
            self.hand_tracker.close()
        cv2.destroyAllWindows()
        pygame.quit()
        print("[Vantage] Closed.")


def main():
    app = VantageApp()
    app.run()


if __name__ == "__main__":
    main()