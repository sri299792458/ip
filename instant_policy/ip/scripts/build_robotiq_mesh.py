import argparse
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


RAW_BASE = "https://raw.githubusercontent.com/PickNikRobotics/ros2_robotiq_gripper/{branch}/robotiq_description"
XACRO_REL = "urdf/robotiq_2f_85_macro.urdf.xacro"


@dataclass
class JointDef:
    name: str
    parent: str
    child: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    mimic_joint: Optional[str]
    mimic_multiplier: float
    mimic_offset: float
    limit_lower: Optional[float]
    limit_upper: Optional[float]


def _format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in vec.tolist()) + "]"


def _parse_vec3(text: Optional[str]) -> np.ndarray:
    if text is None:
        return np.zeros(3, dtype=np.float64)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        raise RuntimeError(f"Expected 3 values, got: {text}")
    return np.asarray(vals, dtype=np.float64)


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    ax = axis / n
    x, y, z = ax.tolist()
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _make_tf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _origin_tf(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return _make_tf(_rpy_to_rot(rpy), xyz)


def _clean_name(name: str) -> str:
    return name.replace("${prefix}", "")


def _strip_package_uri(uri: str) -> str:
    prefix = "package://robotiq_description/"
    if not uri.startswith(prefix):
        raise RuntimeError(f"Unsupported mesh URI (expected {prefix}...): {uri}")
    return uri[len(prefix) :]


def _download_to(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError(f"Failed downloading {url}: HTTP {resp.status}")
        path.write_bytes(resp.read())


def _parse_2f85_description(
    description_dir: Path,
    source: str,
) -> Tuple[Dict[str, str], List[JointDef]]:
    xacro_path = description_dir / XACRO_REL
    if not xacro_path.exists():
        raise RuntimeError(f"Missing xacro file: {xacro_path}")

    tree = ET.parse(str(xacro_path))
    root = tree.getroot()

    link_mesh_rel: Dict[str, str] = {}
    geom_tag = "collision" if source == "collision" else "visual"
    for link in root.findall(".//link"):
        link_name = _clean_name(link.attrib["name"])
        geom_parent = link.find(geom_tag)
        if geom_parent is None:
            continue
        mesh_el = geom_parent.find("./geometry/mesh")
        if mesh_el is None:
            continue
        mesh_uri = mesh_el.attrib.get("filename")
        if not mesh_uri:
            continue
        link_mesh_rel[link_name] = _strip_package_uri(mesh_uri)

    joints: List[JointDef] = []
    for joint in root.findall(".//joint"):
        name = _clean_name(joint.attrib["name"])
        parent = _clean_name(joint.find("parent").attrib["link"])
        child = _clean_name(joint.find("child").attrib["link"])
        joint_type = joint.attrib["type"]

        origin_el = joint.find("origin")
        xyz = _parse_vec3(origin_el.attrib.get("xyz") if origin_el is not None else None)
        rpy = _parse_vec3(origin_el.attrib.get("rpy") if origin_el is not None else None)

        axis_el = joint.find("axis")
        axis = _parse_vec3(axis_el.attrib.get("xyz") if axis_el is not None else "1 0 0")

        mimic_el = joint.find("mimic")
        mimic_joint = None
        mimic_mult = 1.0
        mimic_off = 0.0
        if mimic_el is not None:
            mimic_joint = _clean_name(mimic_el.attrib["joint"])
            mimic_mult = float(mimic_el.attrib.get("multiplier", "1.0"))
            mimic_off = float(mimic_el.attrib.get("offset", "0.0"))

        limit_el = joint.find("limit")
        lower = float(limit_el.attrib["lower"]) if limit_el is not None and "lower" in limit_el.attrib else None
        upper = float(limit_el.attrib["upper"]) if limit_el is not None and "upper" in limit_el.attrib else None

        joints.append(
            JointDef(
                name=name,
                parent=parent,
                child=child,
                joint_type=joint_type,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_mult,
                mimic_offset=mimic_off,
                limit_lower=lower,
                limit_upper=upper,
            )
        )

    if not link_mesh_rel:
        raise RuntimeError("No link meshes parsed from xacro.")
    if not joints:
        raise RuntimeError("No joints parsed from xacro.")
    return link_mesh_rel, joints


def _resolve_primary_joint(joints: List[JointDef]) -> JointDef:
    by_name = {j.name: j for j in joints}
    preferred = "robotiq_85_left_knuckle_joint"
    if preferred in by_name:
        return by_name[preferred]
    for j in joints:
        if j.joint_type in ("revolute", "continuous") and j.mimic_joint is None:
            return j
    raise RuntimeError("No controllable revolute joint found in parsed URDF.")


def _choose_primary_angle(primary: JointDef, state: str, explicit_angle: Optional[float]) -> float:
    if explicit_angle is not None:
        return float(explicit_angle)
    lo = primary.limit_lower if primary.limit_lower is not None else 0.0
    hi = primary.limit_upper if primary.limit_upper is not None else 0.8
    if state == "open":
        return lo
    if state == "closed":
        return hi
    return 0.5 * (lo + hi)


def _joint_angle_map(joints: List[JointDef], primary: JointDef, primary_angle: float) -> Dict[str, float]:
    by_name = {j.name: j for j in joints}
    cache: Dict[str, float] = {primary.name: float(primary_angle)}

    def compute(name: str) -> float:
        if name in cache:
            return cache[name]
        j = by_name[name]
        if j.mimic_joint is not None:
            q = compute(j.mimic_joint) * j.mimic_multiplier + j.mimic_offset
        else:
            q = 0.0
        if j.limit_lower is not None:
            q = max(q, j.limit_lower)
        if j.limit_upper is not None:
            q = min(q, j.limit_upper)
        cache[name] = float(q)
        return cache[name]

    for j in joints:
        compute(j.name)
    return cache


def _fk_link_tfs(joints: List[JointDef], root_link: str, joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    children: Dict[str, List[JointDef]] = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)

    tfs: Dict[str, np.ndarray] = {root_link: np.eye(4, dtype=np.float64)}
    stack = [root_link]
    while stack:
        parent = stack.pop()
        T_parent = tfs[parent]
        for j in children.get(parent, []):
            q = 0.0
            if j.joint_type in ("revolute", "continuous"):
                q = float(joint_angles.get(j.name, 0.0))
            T = _origin_tf(j.origin_xyz, j.origin_rpy)
            if j.joint_type in ("revolute", "continuous"):
                T = T @ _make_tf(_axis_angle_to_rot(j.axis, q), np.zeros(3, dtype=np.float64))
            tfs[j.child] = T_parent @ T
            stack.append(j.child)
    return tfs


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        dumped = mesh.dump(concatenate=False)
        meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No mesh geometry found in scene file: {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type from {path}: {type(mesh)}")
    if mesh.vertices.size == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def _assemble_mesh(
    description_dir: Path,
    source: str,
    primary_angle: float,
    scale: float,
) -> Tuple[trimesh.Trimesh, JointDef]:
    link_mesh_rel, joints = _parse_2f85_description(description_dir, source)
    children = {j.child for j in joints}
    roots = sorted({j.parent for j in joints if j.parent not in children})
    if not roots:
        raise RuntimeError("Could not determine root link from joint graph.")
    root_link = roots[0]

    primary = _resolve_primary_joint(joints)
    q = _joint_angle_map(joints, primary=primary, primary_angle=primary_angle)
    link_tfs = _fk_link_tfs(joints, root_link=root_link, joint_angles=q)

    meshes: List[trimesh.Trimesh] = []
    for link_name, rel_mesh in link_mesh_rel.items():
        mesh_path = description_dir / rel_mesh
        if not mesh_path.exists():
            raise RuntimeError(f"Missing mesh file required by URDF: {mesh_path}")
        if link_name not in link_tfs:
            # Link not connected to root (unexpected for this model), skip explicitly.
            continue
        mesh = _load_mesh(mesh_path)
        mesh.apply_transform(link_tfs[link_name])
        meshes.append(mesh)

    if not meshes:
        raise RuntimeError("No meshes assembled from URDF description.")
    merged = trimesh.util.concatenate(meshes)
    if not np.isclose(scale, 1.0):
        merged.apply_scale(float(scale))
    merged.remove_unreferenced_vertices()
    return merged, primary


def _download_description(branch: str, dst: Path):
    base = RAW_BASE.format(branch=branch)
    xacro_url = f"{base}/{XACRO_REL}"
    xacro_path = dst / XACRO_REL
    _download_to(xacro_path, xacro_url)

    # Parse xacro once to discover exact mesh files referenced by this version.
    link_mesh_rel, _ = _parse_2f85_description(dst, source="collision")
    link_mesh_rel_visual, _ = _parse_2f85_description(dst, source="visual")
    rels = sorted(set(link_mesh_rel.values()) | set(link_mesh_rel_visual.values()))
    for rel in rels:
        _download_to(dst / rel, f"{base}/{rel}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a single Robotiq 2F-85 mesh from URDF kinematics "
            "(standard link+joint assembly), suitable for --gripper_mesh_path."
        )
    )
    parser.add_argument("--out", required=True, help="Output mesh path (.obj/.ply/.stl).")
    parser.add_argument(
        "--source",
        choices=["collision", "visual"],
        default="collision",
        help="Use collision (recommended for proximity) or visual mesh geometry.",
    )
    parser.add_argument(
        "--state",
        choices=["open", "mid", "closed"],
        default="open",
        help="Jaw state used for URDF joint assembly.",
    )
    parser.add_argument(
        "--left-knuckle-angle",
        type=float,
        default=None,
        help="Optional explicit angle (rad) for the left knuckle joint; overrides --state.",
    )
    parser.add_argument(
        "--robotiq-description-dir",
        default=None,
        help=(
            "Optional local path to robotiq_description directory. "
            "If omitted, files are downloaded from PickNikRobotics/ros2_robotiq_gripper."
        ),
    )
    parser.add_argument("--branch", default="main", help="Branch/tag used for auto-download (default: main).")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied after assembly.")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.robotiq_description_dir is not None:
        description_dir = Path(args.robotiq_description_dir).expanduser().resolve()
        if not description_dir.exists():
            raise RuntimeError(f"--robotiq-description-dir does not exist: {description_dir}")
        # Build directly from local description tree.
        _, joints = _parse_2f85_description(description_dir, args.source)
        primary = _resolve_primary_joint(joints)
        primary_angle = _choose_primary_angle(primary, state=args.state, explicit_angle=args.left_knuckle_angle)
        merged, primary = _assemble_mesh(
            description_dir=description_dir,
            source=args.source,
            primary_angle=primary_angle,
            scale=args.scale,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="robotiq_2f85_desc_") as td:
            desc_dir = Path(td) / "robotiq_description"
            _download_description(args.branch, desc_dir)
            _, joints = _parse_2f85_description(desc_dir, args.source)
            primary = _resolve_primary_joint(joints)
            primary_angle = _choose_primary_angle(primary, state=args.state, explicit_angle=args.left_knuckle_angle)
            merged, primary = _assemble_mesh(
                description_dir=desc_dir,
                source=args.source,
                primary_angle=primary_angle,
                scale=args.scale,
            )

    merged.export(str(out_path))
    bounds = merged.bounds
    extents = bounds[1] - bounds[0]
    print(f"Saved URDF-assembled mesh: {out_path}")
    print(f"source: {args.source}")
    print(f"state: {args.state}")
    print(f"{primary.name}_angle_rad: {float(primary_angle):.6f}")
    print(f"bounds_min: {_format_vec(bounds[0])}")
    print(f"bounds_max: {_format_vec(bounds[1])}")
    print(f"extents_m: {_format_vec(extents)}")


if __name__ == "__main__":
    main()
