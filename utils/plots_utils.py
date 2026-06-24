# pip install open3d smplx numpy
import os
import time
import imageio
import numpy as np
import open3d as o3d
from itertools import cycle

try:
    import torch
    import smplx
except Exception:
    torch = None
    smplx = None


def _load_smplx_sequence(smplx_model, npz_path):
    """
    Returns:
      verts: (T, V, 3) float64
      faces: (F, 3) int32
      joints: (T, J, 3) or None
    """
    data = np.load(npz_path, allow_pickle=True)

    betas, poses, trans, exps = data["betas"], data["poses"], data["trans"], data["expressions"]
    n, c = poses.shape[0], poses.shape[1]
    betas = betas.reshape(1, 300)
    betas = np.tile(betas, (n, 1))
    betas = torch.from_numpy(betas).cuda().float()
    poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
    exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
    trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()

    T = 196

    # Make tensors
    def as_t(x, shape=None):
        if x is None:
            return None
        xt = torch.as_tensor(x, dtype=torch.float32)
        if shape and xt.ndim == 1:
            xt = xt.unsqueeze(0).expand(shape)
        return xt

    #     body_model = smplx.create(smplx_model_dir, model_type="smplx", gender=gender, use_pca=False, flat_hand_mean=True)

    # args = {
    #     "betas": as_t(data.get("betas", None)),
    #     "body_pose": as_t(data.get("poses", None)),
    #     "global_orient": as_t(global_orient),
    #     "left_hand_pose": as_t(left_hand_pose),
    #     "right_hand_pose": as_t(right_hand_pose),
    #     "jaw_pose": as_t(jaw_pose),
    #     "leye_pose": as_t(leye_pose),
    #     "reye_pose": as_t(data.get("reye_pose", None)),
    #     "expression": as_t(data.get("expressions", None)),
    #     "transl": as_t(transl),
    # }

    out = smplx_model(
        betas=betas,
        transl=trans,
        expression=exps,
        jaw_pose=poses[:, 66:69],
        global_orient=poses[:, :3],
        body_pose=poses[:, 3 : 21 * 3 + 3],
        left_hand_pose=poses[:, 25 * 3 : 40 * 3],
        right_hand_pose=poses[:, 40 * 3 : 55 * 3],
        return_verts=True,
        return_joints=True,
        leye_pose=poses[:, 69:72],
        reye_pose=poses[:, 72:75],
    )

    # # Ensure all time-dependent are length T; tile if static
    # for k in list(args.keys()):
    #     x = args[k]
    #     if x is None:
    #         continue
    #     if x.shape[0] == 1 and T > 1:
    #         args[k] = x.expand(T, *x.shape[1:])
    #     elif x.shape[0] != T:
    #         raise ValueError(f"{npz_path}: '{k}' has T={x.shape[0]} but expected T={T}")

    # with torch.no_grad():
    #     out = smplx_model(**{k: v for k, v in args.items() if v is not None})
    vt = out.vertices  # (T, V, 3)
    jt = out.joints  # (T, J, 3)
    verts = vt.cpu().numpy()
    joints = jt.cpu().numpy()
    faces = smplx_model.faces

    verts = np.asarray(verts, dtype=np.float64)
    if faces is None:
        raise ValueError(f"{npz_path} does not include 'faces' and none were derivable.")
    faces = np.asarray(faces, dtype=np.int32)
    joints = None if joints is None else np.asarray(joints, dtype=np.float64)
    return verts, faces, joints


def _align_sequence(verts, joints=None, mode="pelvis"):
    """
    Align each frame to the same place.
    mode='pelvis' uses joints[:, 0] if available, else falls back to centroid.
    """
    T = verts.shape[0]
    if mode == "pelvis" and joints is not None:
        roots = joints[:, 0, :]  # SMPL/SMPL-X pelvis is joint 0
    else:
        roots = verts.mean(axis=1)  # centroid per frame
    return verts - roots[:, None, :]


def make_lineset(verts_frame, edges, color=(0, 0, 0)):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(verts_frame, float))
    ls.lines = o3d.utility.Vector2iVector(edges)
    col = np.tile(np.asarray(color, float)[None, :], (edges.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls


def render_offscreen_meshes_over_time(
    meshes_per_frame,
    linesets_per_frame,
    out_mp4="overlay.mp4",
    w=1280,
    h=720,
    fps=30,
    unlit=True,
    azim_deg=45.0,  # rotation around the up-axis (Y)
    elev_deg=20.0,  # elevation above horizontal
    radius_mul=1.0,  # multiply the auto distance
    distance=None,  # OR set an absolute distance (overrides radius_mul)
    fov_deg=50.0,  # field of view (perspective "zoom")
    up=(0, 1, 0),  # world up direction
):
    r = o3d.visualization.rendering.OffscreenRenderer(w, h)
    r.scene.set_background([1, 1, 1, 1])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit" if unlit else "defaultLit"

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"  # line shader
    mat_line.line_width = 1.5  # tweak thickness

    writer = imageio.get_writer(out_mp4, fps=fps)
    try:
        cam_set = False
        for indx_frame, meshes in enumerate(meshes_per_frame):
            if not isinstance(meshes, (list, tuple)):
                meshes = [meshes]

            r.scene.clear_geometry()

            # Track union AABB via min/max vectors (works on all versions)
            aabb_min, aabb_max = None, None

            for i, m in enumerate(meshes):
                r.scene.add_geometry(f"m{i}", m, mat)
                r.scene.add_geometry(f"outline{i}", linesets_per_frame[indx_frame][i], mat_line)
                bb = m.get_axis_aligned_bounding_box()
                lo = np.asarray(bb.get_min_bound(), dtype=np.float64)
                hi = np.asarray(bb.get_max_bound(), dtype=np.float64)
                if aabb_min is None:
                    aabb_min, aabb_max = lo, hi
                else:
                    aabb_min = np.minimum(aabb_min, lo)
                    aabb_max = np.maximum(aabb_max, hi)

            if not cam_set and aabb_min is not None:
                center = (aabb_min + aabb_max) / 2.0
                extent = aabb_max - aabb_min
                base_radius = float(np.linalg.norm(extent)) * 0.8 + 1e-6
                # distance to target
                dist = float(distance) if distance is not None else base_radius * float(radius_mul)

                # spherical to Cartesian (Y is up)
                az = np.deg2rad(azim_deg)
                el = np.deg2rad(elev_deg)
                dir_vec = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)], dtype=float)
                eye = center + dist * dir_vec
                # c = (aabb_min + aabb_max) / 2.0
                # ext = aabb_max - aabb_min
                # radius = float(np.linalg.norm(ext) * 0.7 + 1e-6)
                # eye = c + np.array([radius, radius * 0.5, radius], dtype=np.float64)

                r.scene.camera.look_at(center.tolist(), eye.tolist(), up)

                cam = r.scene.camera
                if hasattr(cam, "set_projection"):
                    aspect = w / float(h)
                    near = max(1e-3, dist * 0.01)
                    far = dist * 10.0 + 1.0
                    # FovType may vary by version; call without if not present
                    try:
                        cam.set_projection(fov_deg, aspect, near, far)
                    except TypeError:
                        # from o3d.visualization.rendering import Camera
                        cam.set_projection(fov_deg, aspect, near, far, o3d.visualization.rendering.Camera.FovType.Vertical)

                cam_set = True

            img = r.render_to_image()
            writer.append_data(np.asarray(img))
    finally:
        writer.close()


def make_mesh(verts_frame, faces, color):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts_frame.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    col = np.tile(np.asarray(color, float)[None, :], (verts_frame.shape[0], 1))
    m.vertex_colors = o3d.utility.Vector3dVector(col)
    m.compute_vertex_normals()
    return m


def faces_to_unique_edges(faces):
    # faces: (F,3) int
    f = np.asarray(faces, dtype=np.int32)
    E = set()
    for a, b, c in f:
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (c, a) if c < a else (a, c)
        E.update((e1, e2, e3))
    return np.asarray(sorted(E), dtype=np.int32)  # (E,2)


def render_smplx_overlay(
    smplx_model,
    npz_paths,
    colors=None,  # ((1.0, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.4, 1.0)),
    fps=30,
    align_mode="pelvis",
    output_video=None,
    frames_dir=None,
):
    """
    Overlay 3 SMPL-X motion sequences in different colors, aligned per-frame.

    Args:
      npz_paths: list/tuple of 3 paths to .npz files
      colors: list of 3 RGB tuples in [0,1]
      smplx_model_dir: path to SMPL-X model folder (needed if npz has params only)
      gender: 'neutral'|'male'|'female' for SMPL-X
      fps: playback rate
      align_mode: 'pelvis' or 'centroid'
    """
    # assert len(npz_paths) == 3, "Please pass exactly 3 .npz paths."
    npz_paths = list(npz_paths)
    assert len(npz_paths) >= 1, "Pass at least one .npz path"
    # Nice distinct palette; will cycle if you have more sequences than colors
    default_palette = [
        (0.90, 0.10, 0.10),
        (0.10, 0.70, 0.10),
        (0.10, 0.30, 0.95),
        (0.90, 0.60, 0.05),
        (0.55, 0.15, 0.80),
        (0.10, 0.75, 0.75),
        (0.60, 0.60, 0.60),
        (0.80, 0.20, 0.50),
        (0.35, 0.70, 0.95),
        (0.50, 0.50, 0.10),
    ]
    if colors is None:
        colors = [c for _, c in zip(range(len(npz_paths)), cycle(default_palette))]
    elif len(colors) < len(npz_paths):
        colors = [c for _, c in zip(range(len(npz_paths)), cycle(colors))]
    meshes = []
    seqs = []
    faces_ref = None
    V_ref = None
    T_min = None

    # Load all sequences
    for p in npz_paths:
        v, f, j = _load_smplx_sequence(smplx_model, p)

        v = _align_sequence(v, j, mode=align_mode)
        if faces_ref is None:
            faces_ref = f
            V_ref = v.shape[1]
        else:
            if v.shape[1] != V_ref:
                raise ValueError(f"Vertex count mismatch: {p} has {v.shape[1]} vs {V_ref}")
            if f.shape != faces_ref.shape or not np.array_equal(f, faces_ref):
                # Topology should match for overlay
                raise ValueError(f"Face topology mismatch for {p}.")
        T_min = v.shape[0] if T_min is None else min(T_min, v.shape[0])
        seqs.append((v, j))

    # Trim to shortest sequence
    seqs = [(v[:T_min], j if j is None else j[:T_min]) for (v, j) in seqs]

    frames = []
    linesets = []
    # Precompute once (topology identical across frames)
    edges = faces_to_unique_edges(faces_ref)
    for t in range(T_min):
        meshes_t = [make_mesh(seqs[i][0][t], faces_ref, colors[i % len(colors)]) for i in range(len(seqs))]
        # for m in meshes_t:
        #     m.compute_vertex_normals()
        #     v = np.asarray(m.vertices)
        #     n = np.asarray(m.vertex_normals)
        #     v_outline = v + 0.002 * n  # tweak epsilon
        #     linesets_t= make_lineset(v_outline, edges, (0,0,0))
        #
        linesets_t = [make_lineset(seqs[i][0][t], edges, (0, 0, 0)) for i in range(len(seqs))]
        linesets.append(linesets_t)
        frames.append(meshes_t)

    # # Build Open3D meshes
    # meshes = []
    # for color in colors:
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(np.zeros((V_ref, 3)))
    #     mesh.triangles = o3d.utility.Vector3iVector(faces_ref)
    #     c = np.tile(np.asarray(color, dtype=np.float64)[None, :], (V_ref, 1))
    #     mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    #     mesh.compute_vertex_normals()
    #     meshes.append(mesh)

    render_offscreen_meshes_over_time(
        frames, linesets, out_mp4=output_video, w=1280, h=720, fps=30, unlit=True, azim_deg=120, elev_deg=15, distance=3.5, fov_deg=30
    )
