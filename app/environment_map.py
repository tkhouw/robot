"""
app/environment_map.py

Parses models/wavego/scene.xml as the single source of truth for the domestic floorplan.
Extracts outer perimeter walls, internal partition walls, solid furniture no-go zones,
and walkable elevated furniture clearances (e.g. dining table and coffee table).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class StaticObstacle:
    """Represents a 2D bounding box obstacle on the floor."""

    def __init__(self, name: str, min_x: float, max_x: float, min_y: float, max_y: float, is_walkable_under: bool = False):
        self.name = name
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.is_walkable_under = is_walkable_under

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            (self.min_x - margin) <= x <= (self.max_x + margin)
            and (self.min_y - margin) <= y <= (self.max_y + margin)
        )


class EnvironmentMap:
    """
    Parses models/wavego/scene.xml directly to ensure that simulation geometry
    and agent coordinate boundaries remain perfectly synchronized.
    """

    def __init__(self, scene_xml_path: Optional[str] = None):
        if scene_xml_path is None:
            # Locate models/wavego/scene.xml relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            scene_xml_path = str(base_dir / "models" / "wavego" / "scene.xml")

        self.scene_path = scene_xml_path
        self.min_x = -1.20
        self.max_x = 5.00
        self.min_y = -2.40
        self.max_y = 2.40
        self.obstacles: List[StaticObstacle] = []
        self._parse_scene_xml()

    def _parse_scene_xml(self):
        try:
            tree = ET.parse(self.scene_path)
            root = tree.getroot()
            worldbody = root.find("worldbody")
            if worldbody is None:
                return

            # 1. Parse perimeter walls & interior walls from worldbody geoms
            min_xs, max_xs = [], []
            min_ys, max_ys = [], []

            for geom in worldbody.findall("geom"):
                name = geom.get("name", "")
                pos_str = geom.get("pos", "")
                size_str = geom.get("size", "")
                if not pos_str or not size_str:
                    continue

                pos = [float(v) for v in pos_str.split()]
                size = [float(v) for v in size_str.split()]

                if name == "wall_west":
                    min_xs.append(pos[0] + size[0])
                elif name.startswith("wall_east"):
                    max_xs.append(pos[0] - size[0])
                elif name == "wall_south":
                    min_ys.append(pos[1] + size[1])
                elif name == "wall_north":
                    max_ys.append(pos[1] - size[1])
                elif name == "kitchen_wall":
                    # Interior partition wall: runs along x=2.10
                    self.obstacles.append(
                        StaticObstacle(
                            name="kitchen_wall",
                            min_x=pos[0] - size[0],
                            max_x=pos[0] + size[0],
                            min_y=pos[1] - size[1],
                            max_y=pos[1] + size[1],
                            is_walkable_under=False,
                        )
                    )

            if min_xs:
                self.min_x = max(min_xs)
            if max_xs:
                self.max_x = min(max_xs)
            if min_ys:
                self.min_y = max(min_ys)
            if max_ys:
                self.max_y = min(max_ys)

            # 2. Parse furniture bodies
            for body in worldbody.findall("body"):
                b_name = body.get("name", "")
                b_pos_str = body.get("pos", "0 0 0")
                b_pos = [float(v) for v in b_pos_str.split()]

                geoms = body.findall("geom")
                # Check if elevated table where robot can walk underneath (e.g. dining table, coffee table)
                has_elevated_top = b_name in ("dining_table", "coffee_table")

                xs, ys = [], []
                for g in geoms:
                    g_pos = [float(v) for v in (g.get("pos") or "0 0 0").split()]
                    g_size = [float(v) for v in (g.get("size") or "0 0 0").split()]
                    g_type = g.get("type", "box")
                    gx = b_pos[0] + g_pos[0]
                    gy = b_pos[1] + g_pos[1]

                    if g_type == "box":
                        xs.extend([gx - g_size[0], gx + g_size[0]])
                        ys.extend([gy - g_size[1], gy + g_size[1]])
                    elif g_type in ("cylinder", "sphere"):
                        r = g_size[0]
                        xs.extend([gx - r, gx + r])
                        ys.extend([gy - r, gy + r])

                if xs and ys:
                    self.obstacles.append(
                        StaticObstacle(
                            name=b_name,
                            min_x=min(xs),
                            max_x=max(xs),
                            min_y=min(ys),
                            max_y=max(ys),
                            is_walkable_under=has_elevated_top,
                        )
                    )
        except Exception as e:
            print(f"[ENVIRONMENT MAP] Warning: Could not parse scene.xml ({e}). Using default floor bounds.")

    def validate_target(self, x: float, y: float, clearance_margin: float = 0.18) -> Tuple[bool, str]:
        """
        Validates if target coordinates (x, y) are safely navigable.
        Checks outer room boundaries and static no-go furniture zones.
        Returns (is_valid, reason_or_suggestion).
        """
        # 1. Check outer walls (allowing clearance_margin from inner wall faces)
        if x < (self.min_x + clearance_margin):
            return False, f"Target x={x:.2f} is too close to or through the West wall (safe min x: {self.min_x + clearance_margin:.2f}m)."
        if x > (self.max_x - clearance_margin):
            return False, f"Target x={x:.2f} is too close to or through the East wall (safe max x: {self.max_x - clearance_margin:.2f}m)."
        if y < (self.min_y + clearance_margin):
            return False, f"Target y={y:.2f} is too close to or through the South wall (safe min y: {self.min_y + clearance_margin:.2f}m)."
        if y > (self.max_y - clearance_margin):
            return False, f"Target y={y:.2f} is too close to or through the North wall (safe max y: {self.max_y - clearance_margin:.2f}m)."

        # 2. Check no-go obstacles
        for obs in self.obstacles:
            if obs.is_walkable_under:
                continue  # Robot can pass under dining/coffee table tops
            if obs.contains(x, y, margin=clearance_margin):
                clean_name = obs.name.replace("_", " ")
                return (
                    False,
                    f"Target ({x:.2f}, {y:.2f}) is inside or too close to the {clean_name}. Please pick a nearby clear floor location.",
                )

        return True, "Valid floor coordinate."

    def get_summary(self) -> Dict[str, any]:
        """Returns a summary of room boundaries and static obstacle zones."""
        return {
            "navigable_range": {
                "x_range": [round(self.min_x + 0.20, 2), round(self.max_x - 0.20, 2)],
                "y_range": [round(self.min_y + 0.20, 2), round(self.max_y - 0.20, 2)],
            },
            "no_go_zones": [
                {
                    "name": obs.name,
                    "x": [round(obs.min_x, 2), round(obs.max_x, 2)],
                    "y": [round(obs.min_y, 2), round(obs.max_y, 2)],
                }
                for obs in self.obstacles
                if not obs.is_walkable_under
            ],
            "walkable_elevated_tables": [
                {
                    "name": obs.name,
                    "x": [round(obs.min_x, 2), round(obs.max_x, 2)],
                    "y": [round(obs.min_y, 2), round(obs.max_y, 2)],
                }
                for obs in self.obstacles
                if obs.is_walkable_under
            ],
        }


# Global singleton instance
_environment_map: Optional[EnvironmentMap] = None


def get_environment_map() -> EnvironmentMap:
    global _environment_map
    if _environment_map is None:
        _environment_map = EnvironmentMap()
    return _environment_map
