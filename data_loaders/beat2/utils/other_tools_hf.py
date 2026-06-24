import os
import numpy as np
import random
import torch
import csv
import pprint
import pandas as pd
from loguru import logger
from collections import OrderedDict
import matplotlib.pyplot as plt
import hashlib
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
from data_loaders.beat2.utils.media import add_audio_to_video
from data_loaders.beat2.utils.fast_render import generate_silent_videos, generate_silent_videos_no_gt
import smplx

from data_loaders.beat2.utils.media import add_subtitles


def write_wav_names_to_csv(folder_path, csv_path):
    """
    Traverse a folder and write the base names of all .wav files to a CSV file.

    :param folder_path: Path to the folder to traverse.
    :param csv_path: Path to the CSV file to write.
    """
    # Open the CSV file for writing
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["id", "type"])

        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if the file ends with .wav
                if file.endswith(".wav"):
                    # Extract the base name without the extension
                    base_name = os.path.splitext(file)[0]
                    # Write the base name and type to the CSV
                    writer.writerow([base_name, "test"])


def resize_motion_sequence_tensor(sequence, target_frames):
    """
    Resize a batch of 8-frame motion sequences to a specified number of frames using interpolation.

    :param sequence: A (bs, 8, 165) tensor representing a batch of 8-frame motion sequences
    :param target_frames: An integer representing the desired number of frames in the output sequences
    :return: A (bs, target_frames, 165) tensor representing the resized motion sequences
    """
    bs, _, _ = sequence.shape

    # Create a time vector for the original and target sequences
    original_time = torch.linspace(0, 1, 8, device=sequence.device).view(1, -1, 1)
    target_time = torch.linspace(0, 1, target_frames, device=sequence.device).view(1, -1, 1)

    # Permute the dimensions to (bs, 165, 8) for interpolation
    sequence = sequence.permute(0, 2, 1)

    # Interpolate each joint's motion to the target number of frames
    resized_sequence = torch.nn.functional.interpolate(sequence, size=target_frames, mode="linear", align_corners=True)

    # Permute the dimensions back to (bs, target_frames, 165)
    resized_sequence = resized_sequence.permute(0, 2, 1)

    return resized_sequence


def adjust_speed_according_to_ratio_tensor(chunks):
    """
    Adjust the playback speed within a batch of 32-frame chunks according to random intervals.

    :param chunks: A (bs, 32, 165) tensor representing a batch of motion chunks
    :return: A (bs, 32, 165) tensor representing the motion chunks after speed adjustment
    """
    bs, _, _ = chunks.shape

    # Step 1: Divide the chunk into 4 equal intervals of 8 frames
    equal_intervals = torch.chunk(chunks, 4, dim=1)

    # Step 2: Randomly sample 3 points within the chunk to determine new intervals
    success = 0
    all_success = []
    # sample_points = torch.sort(torch.randint(1, 32, (bs, 3), device=chunks.device), dim=1).values
    # new_intervals_boundaries = torch.cat([torch.zeros((bs, 1), device=chunks.device, dtype=torch.long), sample_points, 32*torch.ones((bs, 1), device=chunks.device, dtype=torch.long)], dim=1)
    while success != 1:
        sample_points = sorted(random.sample(range(1, 32), 3))
        new_intervals_boundaries = [0] + sample_points + [32]
        new_intervals = [chunks[0][new_intervals_boundaries[i] : new_intervals_boundaries[i + 1]] for i in range(4)]
        speed_ratios = [8 / len(new_interval) for new_interval in new_intervals]
        # if any of the speed ratios is greater than 3 or less than 0.33, resample
        if all([0.33 <= speed_ratio <= 3 for speed_ratio in speed_ratios]):
            success += 1
            all_success.append(new_intervals_boundaries)
    new_intervals_boundaries = torch.from_numpy(np.array(all_success))
    all_shapes = new_intervals_boundaries[:, 1:] - new_intervals_boundaries[:, :-1]
    # Step 4: Adjust the speed of each new interval
    adjusted_intervals = []
    for i in range(4):
        adjusted_interval = resize_motion_sequence_tensor(equal_intervals[i], all_shapes[0, i])
        adjusted_intervals.append(adjusted_interval)

    # Step 5: Concatenate the adjusted intervals
    adjusted_chunk = torch.cat(adjusted_intervals, dim=1)

    return adjusted_chunk


def compute_exact_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def compute_iou(mask1, mask2):
    # Compute the intersection
    intersection = np.logical_and(mask1, mask2).sum()

    # Compute the union
    union = np.logical_or(mask1, mask2).sum()

    # Compute the IoU
    iou = intersection / union

    return iou


def blankblending(all_frames, x, n):
    return all_frames[x : x + n + 1]


def synthesize_intermediate_frames_FILM(frame1, frame2, t, name, save_path):
    import replicate
    from urllib.request import urlretrieve
    import os

    cv2.imwrite(save_path[:-9] + name + "_frame1.png", frame1)
    cv2.imwrite(save_path[:-9] + name + "_frame2.png", frame2)
    os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN", "")
    output = replicate.run(
        "google-research/frame-interpolation:4f88a16a13673a8b589c18866e540556170a5bcb2ccdc12de556e800e9456d3d",
        input={
            "frame1": open(save_path[:-9] + name + "_frame1.png", "rb"),
            "frame2": open(save_path[:-9] + name + "_frame2.png", "rb"),
            "times_to_interpolate": t,
        },
    )
    print(output)
    urlretrieve(output, save_path[:-9] + name + "_inter.mp4")
    return load_video_as_numpy_array(save_path[:-9] + name + "_inter.mp4")


def load_video_as_numpy_array(video_path):
    cap = cv2.VideoCapture(video_path)

    # Using list comprehension to read frames and store in a list
    frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None)) if ret]

    cap.release()

    return np.array(frames)


def synthesize_intermediate_frames_bidirectional(all_frames, x, n):
    frame1 = all_frames[x]
    frame2 = all_frames[x + n]

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the forward and backward optical flow
    forward_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward_flow = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    synthesized_frames = []
    for i in range(1, n):  # For each intermediate frame between x and x + n
        alpha = i / n  # Interpolation factor

        # Compute the intermediate forward and backward flow
        intermediate_forward_flow = forward_flow * alpha
        intermediate_backward_flow = backward_flow * (1 - alpha)

        # Warp the frames based on the intermediate flow
        h, w = frame1.shape[:2]
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
        forward_displacement = flow_map + intermediate_forward_flow.reshape(-1, 2)
        backward_displacement = flow_map - intermediate_backward_flow.reshape(-1, 2)

        # Use cv2.remap for efficient warping
        remap_x_forward, remap_y_forward = np.clip(forward_displacement[:, 1], 0, w - 1), np.clip(forward_displacement[:, 0], 0, h - 1)
        remap_x_backward, remap_y_backward = np.clip(backward_displacement[:, 1], 0, w - 1), np.clip(backward_displacement[:, 0], 0, h - 1)

        warped_forward = cv2.remap(
            frame1, remap_x_forward.reshape(h, w).astype(np.float32), remap_y_forward.reshape(h, w).astype(np.float32), interpolation=cv2.INTER_LINEAR
        )
        warped_backward = cv2.remap(
            frame2,
            remap_x_backward.reshape(h, w).astype(np.float32),
            remap_y_backward.reshape(h, w).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
        )

        # Blend the warped frames to generate the intermediate frame
        intermediate_frame = cv2.addWeighted(warped_forward, 1 - alpha, warped_backward, alpha, 0)
        synthesized_frames.append(intermediate_frame)

    return synthesized_frames  # Return n-2 synthesized intermediate frames


def linear_interpolate_frames(all_frames, x, n):
    frame1 = all_frames[x]
    frame2 = all_frames[x + n]

    synthesized_frames = []
    for i in range(1, n):  # For each intermediate frame between x and x + n
        alpha = i / (n)  # Correct interpolation factor
        inter_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        synthesized_frames.append(inter_frame)
    return synthesized_frames[:-1]


def warp_frame(src_frame, flow):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    displacement = flow_map + flow.reshape(-1, 2)

    # Extract x and y coordinates of the displacement
    x_coords = np.clip(displacement[:, 1], 0, w - 1).reshape(h, w).astype(np.float32)
    y_coords = np.clip(displacement[:, 0], 0, h - 1).reshape(h, w).astype(np.float32)

    # Use cv2.remap for efficient warping
    warped_frame = cv2.remap(src_frame, x_coords, y_coords, interpolation=cv2.INTER_LINEAR)

    return warped_frame


def synthesize_intermediate_frames(all_frames, x, n):
    # Calculate Optical Flow between the first and last frame
    frame1 = cv2.cvtColor(all_frames[x], cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(all_frames[x + n], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    synthesized_frames = []
    for i in range(1, n):  # For each intermediate frame
        alpha = i / (n)  # Interpolation factor
        intermediate_flow = flow * alpha  # Interpolate the flow
        intermediate_frame = warp_frame(all_frames[x], intermediate_flow)  # Warp the first frame
        synthesized_frames.append(intermediate_frame)

    return synthesized_frames


def map2color(s):
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    color_code = m.hexdigest()[:6]
    return "#" + color_code


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def adjust_array(x, k):
    len_x = len(x)
    len_k = len(k)

    # If x is shorter than k, pad with zeros
    if len_x < len_k:
        return np.pad(x, (0, len_k - len_x), "constant")

    # If x is longer than k, truncate x
    elif len_x > len_k:
        return x[:len_k]

    # If both are of same length
    else:
        return x


def onset_to_frame(onset_times, audio_length, fps):
    # Calculate total number of frames for the given audio length
    total_frames = int(audio_length * fps)

    # Create an array of zeros of shape (total_frames,)
    frame_array = np.zeros(total_frames, dtype=np.int32)

    # For each onset time, calculate the frame number and set it to 1
    for onset in onset_times:
        frame_num = int(onset * fps)
        # Check if the frame number is within the array bounds
        if 0 <= frame_num < total_frames:
            frame_array[frame_num] = 1

    return frame_array


def smooth_rotvec_animations(animation1, animation2, blend_frames):
    """
    Smoothly transition between two animation clips using SLERP.

    Parameters:
    - animation1: The first animation clip, a numpy array of shape [n, k].
    - animation2: The second animation clip, a numpy array of shape [n, k].
    - blend_frames: Number of frames over which to blend the two animations.

    Returns:
    - A smoothly blended animation clip of shape [2n, k].
    """

    # Ensure blend_frames doesn't exceed the length of either animation
    n1, k1 = animation1.shape
    n2, k2 = animation2.shape
    animation1 = animation1.reshape(n1, k1 // 3, 3)
    animation2 = animation2.reshape(n2, k2 // 3, 3)
    blend_frames = min(blend_frames, len(animation1), len(animation2))
    all_int = []
    for i in range(k1 // 3):
        # Convert rotation vectors to quaternion for the overlapping part
        q = R.from_rotvec(np.concatenate([animation1[0:1, i], animation2[-2:-1, i]], axis=0))  # .as_quat()
        # q2 = R.from_rotvec()#.as_quat()
        times = [0, blend_frames * 2 - 1]
        slerp = Slerp(times, q)
        interpolated = slerp(np.arange(blend_frames * 2))
        interpolated_rotvecs = interpolated.as_rotvec()
        all_int.append(interpolated_rotvecs)
    interpolated_rotvecs = np.concatenate(all_int, axis=1)
    # result = np.vstack((animation1[:-blend_frames], interpolated_rotvecs, animation2[blend_frames:]))
    result = interpolated_rotvecs.reshape(2 * n1, k1)
    return result


def smooth_animations(animation1, animation2, blend_frames):
    """
    Smoothly transition between two animation clips using linear interpolation.

    Parameters:
    - animation1: The first animation clip, a numpy array of shape [n, k].
    - animation2: The second animation clip, a numpy array of shape [n, k].
    - blend_frames: Number of frames over which to blend the two animations.

    Returns:
    - A smoothly blended animation clip of shape [2n, k].
    """

    # Ensure blend_frames doesn't exceed the length of either animation
    blend_frames = min(blend_frames, len(animation1), len(animation2))

    # Extract overlapping sections
    overlap_a1 = animation1[-blend_frames : -blend_frames + 1, :]
    overlap_a2 = animation2[blend_frames - 1 : blend_frames, :]

    # Create blend weights for linear interpolation
    alpha = np.linspace(0, 1, 2 * blend_frames).reshape(-1, 1)

    # Linearly interpolate between overlapping sections
    blended_overlap = overlap_a1 * (1 - alpha) + overlap_a2 * alpha

    # Extend the animations to form the result with 2n frames
    if blend_frames == len(animation1) and blend_frames == len(animation2):
        result = blended_overlap
    else:
        before_blend = animation1[:-blend_frames]
        after_blend = animation2[blend_frames:]
        result = np.vstack((before_blend, blended_overlap, after_blend))
    return result


def interpolate_sequence(quaternions):
    bs, n, j, _ = quaternions.shape
    new_n = 2 * n
    new_quaternions = torch.zeros((bs, new_n, j, 4), device=quaternions.device, dtype=quaternions.dtype)

    for i in range(n):
        q1 = quaternions[:, i, :, :]
        new_quaternions[:, 2 * i, :, :] = q1

        if i < n - 1:
            q2 = quaternions[:, i + 1, :, :]
            new_quaternions[:, 2 * i + 1, :, :] = slerp(q1, q2, 0.5)
        else:
            # For the last point, duplicate the value
            new_quaternions[:, 2 * i + 1, :, :] = q1

    return new_quaternions


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def slerp(q1, q2, t):
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    flip = (dot < 0).float()
    q2 = (1 - flip * 2) * q2
    dot = dot * (1 - flip * 2)

    DOT_THRESHOLD = 0.9995
    mask = (dot > DOT_THRESHOLD).float()

    theta_0 = torch.acos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 = q3 / torch.norm(q3, dim=-1, keepdim=True)

    interpolated = torch.cos(theta) * q1 + torch.sin(theta) * q3

    return mask * (q1 + t * (q2 - q1)) + (1 - mask) * interpolated


def estimate_linear_velocity(data_seq, dt):
    """
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    """
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def velocity2position(data_seq, dt, init_pos):
    res_trans = []
    for i in range(data_seq.shape[1]):
        if i == 0:
            res_trans.append(init_pos.unsqueeze(1))
        else:
            res = data_seq[:, i - 1 : i] * dt + res_trans[-1]
            res_trans.append(res)
    return torch.cat(res_trans, dim=1)


def estimate_angular_velocity(rot_seq, dt):
    """
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    """
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w


def image_from_bytes(image_bytes):
    import matplotlib.image as mpimg
    from io import BytesIO

    return mpimg.imread(BytesIO(image_bytes), format="PNG")


def process_frame(i, vertices_all, vertices1_all, faces, output_dir, filenames):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    import pyrender

    def deg_to_rad(degrees):
        return degrees * np.pi / 180

    uniform_color = [220, 220, 220, 255]
    resolution = (1000, 1000)
    figsize = (10, 10)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figsize[0] * 2, figsize[1] * 1))
    axs = axs.flatten()

    vertices = vertices_all[i]
    vertices1 = vertices1_all[i]
    filename = f"{output_dir}frame_{i}.png"
    filenames.append(filename)
    if i % 100 == 0:
        print("processed", i, "frames")
    # time_s = time.time()
    angle_rad = deg_to_rad(-2)
    pose_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
            [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    angle_rad = deg_to_rad(-30)
    pose_light = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    for vtx_idx, vtx in enumerate([vertices, vertices1]):
        trimesh_mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=uniform_color)
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        scene.add(camera, pose=pose_camera)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(light, pose=pose_light)
        renderer = pyrender.OffscreenRenderer(*resolution)
        color, _ = renderer.render(scene)
        axs[vtx_idx].imshow(color)
        axs[vtx_idx].axis("off")
        renderer.delete()

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def generate_images(frames, vertices_all, vertices1_all, faces, output_dir, filenames):
    import multiprocessing

    # import trimesh
    num_cores = multiprocessing.cpu_count() - 1  # This will get the number of cores on your machine.
    # mesh = trimesh.Trimesh(vertices_all[0], faces)
    # scene = mesh.scene()
    # fov = scene.camera.fov.copy()
    # fov[0] = 80.0
    # fov[1] = 60.0
    # camera_params = {
    #     'fov': fov,
    #     'resolution': scene.camera.resolution,
    #     'focal': scene.camera.focal,
    #     'z_near': scene.camera.z_near,
    #     "z_far": scene.camera.z_far,
    #     'transform': scene.graph[scene.camera.name][0]
    # }
    # mesh1 = trimesh.Trimesh(vertices1_all[0], faces)
    # scene1 = mesh1.scene()
    # camera_params1 = {
    #     'fov': fov,
    #     'resolution': scene1.camera.resolution,
    #     'focal': scene1.camera.focal,
    #     'z_near': scene1.camera.z_near,
    #     "z_far": scene1.camera.z_far,
    #     'transform': scene1.graph[scene1.camera.name][0]
    # }
    # Use a Pool to manage the processes
    # for i in range(frames):
    #     process_frame(i, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames, camera_params, camera_params1)
    for i in range(frames):
        process_frame(i * 3, vertices_all, vertices1_all, faces, output_dir, filenames)

    # progress = multiprocessing.Value('i', 0)
    # lock = multiprocessing.Lock()
    # with multiprocessing.Pool(num_cores) as pool:
    #     # pool.starmap(process_frame, [(i, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames, camera_params, camera_params1) for i in range(frames)])
    #     pool.starmap(
    #         process_frame,
    #         [
    #             (i, vertices_all, vertices1_all, faces, output_dir, filenames)
    #             for i in range(frames)
    #         ]
    #     )

    # progress = multiprocessing.Value('i', 0)
    # lock = multiprocessing.Lock()
    # with multiprocessing.Pool(num_cores) as pool:
    #     # pool.starmap(process_frame, [(i, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames, camera_params, camera_params1) for i in range(frames)])
    #     pool.starmap(
    #         process_frame,
    #         [
    #             (i, vertices_all, vertices1_all, faces, output_dir, filenames)
    #             for i in range(frames)
    #         ]
    #     )


def rotate_vertices(vertices, deg=50.0, axis="y", center="centroid", shift=[0, 0, 0]):
    """
    vertices: np.ndarray (T, V, 3) or (V, 3)
    deg: rotation in degrees
    axis: 'x' | 'y' | 'z'
    center: 'origin' | 'centroid' | np.ndarray broadcastable to (T,1,3)
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim == 2:
        verts = verts[None, ...]  # (1,V,3) for unified handling

    T, V, _ = verts.shape
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    elif axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # rotation center
    if center == "origin":
        C = np.zeros((T, 1, 3), dtype=np.float64)
    elif center == "centroid":
        C = verts.mean(axis=1, keepdims=True)  # per-frame centroid
    else:
        C = np.asarray(center, dtype=np.float64)
        if C.ndim == 1:
            C = C[None, None, :]  # (1,1,3)
        if C.shape == (T, 3):
            C = C[:, None, :]  # (T,1,3)
        assert C.shape == (T, 1, 3), f"center shape must broadcast to (T,1,3); got {C.shape}"

    # apply rotation: (T,V,3)
    verts_rot = np.einsum("ij,tfj->tfi", R, verts - C) + C + shift
    return verts_rot if vertices.ndim == 3 else verts_rot[0]


def render_overlay_sequence(
    res_npz_path, gt_npz_path, output_dir, args, smplx_params: dict = None, smplx_model=None, device="cpu", audio_path=None, name="", keyframes=None
):

    # Create smplx model
    if smplx_model is None:
        assert smplx_params is not None, "Define smplx parameters"
        smplx_model = smplx.create(**smplx_params).to(device)

    # Load data
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    gt_np_body = np.load(gt_npz_path, allow_pickle=True)

    # Create folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get smplx faces
    faces = np.load(f"{smplx_params['model_path']}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    # Get prediction data from npz and create smplx sequence
    n = data_np_body["poses"].shape[0]
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).to(device)
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).to(device)
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).to(device)
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).to(device)
    output = smplx_model(
        betas=beta,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=pose[:, :3],
        body_pose=pose[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose[:, 40 * 3 : 55 * 3],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        return_verts=True,
    )
    vertices_all = output["vertices"].cpu().detach().numpy()
    # normilize
    average = np.mean(vertices_all.reshape(-1, 3), axis=0)
    vertices_all = vertices_all - average + np.array([0, 1.2, 0])
    vertices_all_rot_right = rotate_vertices(vertices_all, deg=40, axis="y", center="centroid")
    vertices_all_rot_left = rotate_vertices(vertices_all, deg=-40, axis="y", center="centroid")
    vertices_all_shift_left = rotate_vertices(vertices_all, deg=0, axis="y", center="centroid", shift=[-0.5, 0, 0])

    # Get gt data from npz and create smplx sequence
    beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    expression1 = torch.from_numpy(gt_np_body["expressions"][:n]).to(torch.float32).to(device)
    jaw_pose1 = torch.from_numpy(gt_np_body["poses"][:n, 66:69]).to(torch.float32).to(device)
    pose1 = torch.from_numpy(gt_np_body["poses"][:n]).to(torch.float32).to(device)
    transl1 = torch.from_numpy(gt_np_body["trans"][:n]).to(torch.float32).to(device)
    output1 = smplx_model(
        betas=beta1,
        transl=transl1,
        expression=expression1,
        jaw_pose=jaw_pose1,
        global_orient=pose1[:, :3],
        body_pose=pose1[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose1[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose1[:, 40 * 3 : 55 * 3],
        leye_pose=pose1[:, 69:72],
        reye_pose=pose1[:, 72:75],
        return_verts=True,
    )
    vertices1_all = output1["vertices"].cpu().detach().numpy()

    # average = np.mean(vertices1_all.reshape(-1, 3), axis=0)
    vertices1_all = vertices1_all - average + np.array([0, 1.2, 0])
    vertices_all1_rot_right = rotate_vertices(vertices1_all, deg=40, axis="y", center="centroid")
    vertices_all1_rot_left = rotate_vertices(vertices1_all, deg=-40, axis="y", center="centroid")
    vertices_all1_shift_right = rotate_vertices(vertices1_all, deg=0, axis="y", center="centroid", shift=[0.5, 0, 0])

    # Generate silent video
    seconds = vertices_all.shape[0] // 30

    # keyframes_set = np.zeros(int(vertices_all.shape[0])).tolist()
    # if keyframes is not None:
    #     for k in keyframes:
    #         keyframes_set[k] = 1
    #         if k != 0:
    #             keyframes_set[k - 1] = 1
    #         if k != len(keyframes_set) - 1:
    #             keyframes_set[k + 1] = 1

    silent_video_file_path = generate_silent_videos(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        2,  # args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        [vertices_all_rot_right, vertices_all_shift_left, vertices_all_rot_left],
        [vertices_all1_rot_right, vertices_all1_shift_right, vertices_all1_rot_left],
        faces,
        output_dir,
        name,
        # keyframes_set,
        device_id=device,
        overlay=True,
    )

    # Add audio too video
    if audio_path is not None:
        base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
        final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
    else:
        final_clip = silent_video_file_path

    return final_clip


def postprocess_silent_video(silent_video_file_path, subtitles, audio_data, pose_length=90, audio_sr=16000, pose_fps=30):
    out_path = os.path.dirname(silent_video_file_path)
    base_filename_without_ext = os.path.splitext(os.path.basename(silent_video_file_path))[0]
    if subtitles is not None:

        subtitled_video_file_path = os.path.join(out_path, f"{base_filename_without_ext}_with_sub.mp4")
        add_subtitles(silent_video_file_path, subtitled_video_file_path, subtitles)
    else:
        subtitled_video_file_path = silent_video_file_path
    # os.remove(silent_video_file_path)

    # Add audio to video
    # audio_data = audio_data[: int(np.floor(n / pose_fps * audio_sr))]
    base_filename_without_ext = os.path.splitext(os.path.basename(silent_video_file_path))[0]
    audio_video_file_path = os.path.join(out_path, f"{base_filename_without_ext}.mp4")
    add_audio_to_video(subtitled_video_file_path, audio_video_file_path, audio_data, audio_sr)
    os.remove(subtitled_video_file_path)
    # os.remove(silent_video_file_path)
    print(audio_video_file_path)


def render_one_sequence(
    res_npz_path,
    gt_npz_path,
    output_dir,
    args,
    smplx_params: dict = None,
    smplx_model=None,
    device="cpu",
    audio_path=None,
    name="",
    keyframes=None,
    titles=None,
    overlay=False,
    start_frame=0,
    end_frame=None,
):

    # Create smplx model
    if smplx_model is None:
        assert smplx_params is not None, "Define smplx parameters"
        smplx_model = smplx.create(**smplx_params).to(device)

    # Load data
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    gt_np_body = np.load(gt_npz_path, allow_pickle=True)

    # Create folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get smplx faces
    faces = np.load(f"{smplx_params['model_path']}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    # Get prediction data from npz and create smplx sequence
    if end_frame is not None:
        n = end_frame - start_frame
    else:
        # n = data_np_body["poses"].shape[0]
        n = int(np.min([gt_np_body["poses"].shape[0], data_np_body["poses"].shape[0]]))
        end_frame = n
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][start_frame:end_frame]).to(torch.float32).to(device)
    jaw_pose = torch.from_numpy(data_np_body["poses"][start_frame:end_frame, 66:69]).to(torch.float32).to(device)
    pose = torch.from_numpy(data_np_body["poses"][start_frame:end_frame]).to(torch.float32).to(device)
    transl = torch.from_numpy(data_np_body["trans"][start_frame:end_frame]).to(torch.float32).to(device)
    output = smplx_model(
        betas=beta,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=pose[:, :3],
        body_pose=pose[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose[:, 40 * 3 : 55 * 3],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        return_verts=True,
    )
    vertices_all = output["vertices"].cpu().detach().numpy()
    # Get gt data from npz and create smplx sequence
    beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    expression1 = torch.from_numpy(gt_np_body["expressions"][start_frame:end_frame]).to(torch.float32).to(device)
    jaw_pose1 = torch.from_numpy(gt_np_body["poses"][start_frame:end_frame, 66:69]).to(torch.float32).to(device)
    pose1 = torch.from_numpy(gt_np_body["poses"][start_frame:end_frame]).to(torch.float32).to(device)
    transl1 = torch.from_numpy(gt_np_body["trans"][start_frame:end_frame]).to(torch.float32).to(device)
    output1 = smplx_model(
        betas=beta1,
        transl=transl1,
        expression=expression1,
        jaw_pose=jaw_pose1,
        global_orient=pose1[:, :3],
        body_pose=pose1[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose1[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose1[:, 40 * 3 : 55 * 3],
        leye_pose=pose1[:, 69:72],
        reye_pose=pose1[:, 72:75],
        return_verts=True,
    )
    vertices1_all = output1["vertices"].cpu().detach().numpy()

    # Generate silent video
    seconds = vertices_all.shape[0] // 30

    keyframes_set = np.zeros(int(vertices_all.shape[0])).tolist()
    if keyframes is not None:
        for k in keyframes:
            keyframes_set[k] = 1
            if k != 0:
                keyframes_set[k - 1] = 1
            if k != len(keyframes_set) - 1:
                keyframes_set[k + 1] = 1

    silent_video_file_path = generate_silent_videos(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        2,  # args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        vertices_all,
        vertices1_all,
        faces,
        output_dir,
        name,
        keyframes_set,
        device,
        overlay=overlay,
        titles=titles,
    )

    # silent_video_file_path = generate_silent_videos(
    #     args.render_video_fps,
    #     args.render_video_width,
    #     args.render_video_height,
    #     2,  # args.render_concurrent_num,
    #     args.render_tmp_img_filetype,
    #     int(seconds * args.render_video_fps),
    #     vertices_all,
    #     vertices1_all,
    #     faces,
    #     output_dir,
    #     name,
    #     keyframes_set,
    #     device,
    #     overlay=overlay,
    #     titles=titles,
    # )
    # Add audio too video
    if audio_path is not None:
        base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
        final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
    else:
        final_clip = silent_video_file_path

    return final_clip


def render_one_sequence_res_npz_only(
    res_npz_path,
    output_dir,
    args,
    smplx_params: dict = None,
    smplx_model=None,
    device="cpu",
    audio_path=None,
    name="",
    keyframes=None,
    start_frame=0,
    end_frame=None,
):
    """
    Simplified version of render_one_sequence that only uses res_npz_path (no ground truth).

    Args:
        res_npz_path: Path to the result npz file
        output_dir: Output directory for the video
        args: Arguments object with render settings
        smplx_params: Dictionary of SMPLX model parameters
        smplx_model: Optional pre-created SMPLX model
        device: Device to run on
        audio_path: Optional path to audio file to add to video
        name: Name for the output video
        keyframes: Optional list of keyframe indices
        start_frame: Starting frame index
        end_frame: Ending frame index (None for all frames)

    Returns:
        Path to the generated video file
    """
    # Create smplx model
    if smplx_model is None:
        assert smplx_params is not None, "Define smplx parameters"
        smplx_model = smplx.create(**smplx_params).to(device)

    # Load data
    data_np_body = np.load(res_npz_path, allow_pickle=True)

    # Create folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get smplx faces
    faces = np.load(f"{smplx_params['model_path']}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    # Get prediction data from npz and create smplx sequence
    if end_frame is not None:
        n = end_frame - start_frame
    else:
        n = data_np_body["poses"].shape[0]
        end_frame = n

    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][start_frame:end_frame]).to(torch.float32).to(device)
    jaw_pose = torch.from_numpy(data_np_body["poses"][start_frame:end_frame, 66:69]).to(torch.float32).to(device)
    pose = torch.from_numpy(data_np_body["poses"][start_frame:end_frame]).to(torch.float32).to(device)
    transl = torch.from_numpy(data_np_body["trans"][start_frame:end_frame]).to(torch.float32).to(device)

    output = smplx_model(
        betas=beta,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=pose[:, :3],
        body_pose=pose[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose[:, 40 * 3 : 55 * 3],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        return_verts=True,
    )
    vertices_all = output["vertices"].cpu().detach().numpy()

    # Generate silent video
    seconds = vertices_all.shape[0] // 30

    silent_video_file_path = generate_silent_videos_no_gt(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        2,  # args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        vertices_all,
        faces,
        output_dir,
    )

    # Rename the output file if name is provided
    # generate_silent_videos_no_gt always outputs to "silence_video.mp4"
    if name:
        named_video_path = os.path.join(output_dir, f"{name}_silence_video.mp4")
        if os.path.exists(silent_video_file_path) and silent_video_file_path != named_video_path:
            if os.path.exists(named_video_path):
                os.remove(named_video_path)
            os.rename(silent_video_file_path, named_video_path)
            silent_video_file_path = named_video_path

    # Add audio to video
    if audio_path is not None:
        base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
        final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
    else:
        final_clip = silent_video_file_path

    return final_clip


def plot_pose_xy(pose_xyz, save_path=None, title="Pose", projection_axes=("x", "y"), highlight_joints=None):
    """
    Plot a single pose given as (J, 3) points using SMPL-X kinematic tree for bones.

    Args:
        pose_xyz: numpy array or tensor of shape (J, 3)
        save_path: optional file path to save the plot
        title: figure title
        projection_axes: tuple of axes to project (e.g., ("x","y"), ("x","z"), ("y","z"))
        highlight_joints: list of joint indices to highlight in a different color

    Returns:
        (fig, ax)
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    import numpy as np

    if hasattr(pose_xyz, "detach"):
        pose_xyz = pose_xyz.detach().cpu().numpy()
    pose_xyz = _np.asarray(pose_xyz)
    assert pose_xyz.ndim == 2 and pose_xyz.shape[1] == 3, "pose_xyz must be (J, 3)"

    # Load parents from SMPLX kinematic tree
    smpl_fname = "./datasets/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
    smpl_data = np.load(smpl_fname, encoding="latin1")
    parents = smpl_data["kintree_table"][0].astype(np.int32)

    ax_idx = {"x": 0, "y": 1, "z": 2}
    a0, a1 = ax_idx[projection_axes[0]], ax_idx[projection_axes[1]]

    fig = _plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Prepare joint colors
    J = pose_xyz.shape[0]
    joint_colors = ["lightgray"] * J
    joint_sizes = [15] * J

    if highlight_joints is not None:
        for joint_idx in highlight_joints:
            if 0 <= joint_idx < J:
                joint_colors[joint_idx] = "red"
                joint_sizes[joint_idx] = 30

    # Scatter joints with different colors
    for j in range(J):
        ax.scatter(pose_xyz[j, a0], pose_xyz[j, a1], c=joint_colors[j], s=joint_sizes[j], zorder=3, alpha=0.8)

    # Draw bones
    for j in range(1, J):
        p = parents[j]
        if p < 0 or p >= J:
            continue
        x0, y0 = pose_xyz[p, a0], pose_xyz[p, a1]
        x1, y1 = pose_xyz[j, a0], pose_xyz[j, a1]

        # Use different line colors for highlighted joints
        line_color = "#FF6B6B" if (j in highlight_joints or p in highlight_joints) else "#555555"
        line_width = 2.5 if (j in highlight_joints or p in highlight_joints) else 1.5

        ax.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, zorder=2)

    ax.set_xlabel(projection_axes[0])
    ax.set_ylabel(projection_axes[1])
    ax.set_title(title)
    ax.invert_yaxis()

    # Add legend if highlighting joints
    # if highlight_joints is not None:
    #     from matplotlib.lines import Line2D

    # legend_elements = [
    #     Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Active Joints"),
    #     Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=6, label="Other Joints"),
    # ]
    # ax.legend(handles=legend_elements, loc="upper right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"Saved pose image to {save_path}")

    return fig, ax


def render_one_sequence_no_gt(
    res_npz_path,
    output_dir,
    audio_path,
    model_folder="./datasets/hub/smplx_models/",
    model_type="smplx",
    gender="NEUTRAL_2020",
    ext="npz",
    num_betas=300,
    num_expression_coeffs=100,
    use_face_contour=False,
    use_matplotlib=False,
    args=None,
):
    import smplx
    import matplotlib.pyplot as plt
    import imageio
    from tqdm import tqdm
    import os
    import numpy as np
    import torch
    import moviepy.editor as mp
    import librosa

    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext,
        use_pca=False,
    ).to(device)

    # data_npz = np.load(f"{output_dir}{res_npz_path}.npz")
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    # gt_np_body = np.load(gt_npz_path, allow_pickle=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not use_matplotlib:
    #    import trimesh
    # import pyrender
    from pyvirtualdisplay import Display

    #'''
    # display = Display(visible=0, size=(1000, 1000))
    # display.start()
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    seconds = 1
    # data_npz["jaw_pose"].shape[0]
    n = data_np_body["poses"].shape[0]
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).to(device)
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).to(device)
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).to(device)
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).to(device)
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).to(device)
    output = model(
        betas=beta,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=pose[:, :3],
        body_pose=pose[:, 3 : 21 * 3 + 3],
        left_hand_pose=pose[:, 25 * 3 : 40 * 3],
        right_hand_pose=pose[:, 40 * 3 : 55 * 3],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        return_verts=True,
    )
    vertices_all = output["vertices"].cpu().detach().numpy()

    # beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    # expression1 = torch.from_numpy(gt_np_body["expressions"][:n]).to(torch.float32).cuda()
    # jaw_pose1 = torch.from_numpy(gt_np_body["poses"][:n,66:69]).to(torch.float32).cuda()
    # pose1 = torch.from_numpy(gt_np_body["poses"][:n]).to(torch.float32).cuda()
    # transl1 = torch.from_numpy(gt_np_body["trans"][:n]).to(torch.float32).cuda()
    # output1 = model(betas=beta1, transl=transl1, expression=expression1, jaw_pose=jaw_pose1, global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3],
    #     leye_pose=pose1[:, 69:72],
    #     reye_pose=pose1[:, 72:75],return_verts=True)
    # vertices1_all = output1["vertices"].cpu().detach().numpy()
    if args.debug:
        seconds = 1
    else:
        seconds = vertices_all.shape[0] // 30
    silent_video_file_path = utils.fast_render.generate_silent_videos_no_gt(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        vertices_all,
        faces,
        output_dir,
    )
    base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
    utils.media.add_audio_to_video(silent_video_file_path, audio_path, final_clip)
    os.remove(silent_video_file_path)
    return final_clip


def render_three_sequence(
    base_npz_path,
    res_npz_path,
    tar_npz_path,
    output_dir,
    args,
    smplx_params: dict = None,
    smplx_model=None,
    device="cpu",
    audio_path=None,
    name="",
    titles=None,
):
    """Render a three-panel video: base (left), result (middle), target semantic (right).

    base_npz_path: path to NPZ for base model output (no integration)
    res_npz_path:  path to NPZ for final result (with integration)
    tar_npz_path:  path to NPZ for target semantic motion
    titles: optional list of three strings for [left, middle, right]
    """

    # Create smplx model
    if smplx_model is None:
        assert smplx_params is not None, "Define smplx parameters"
        smplx_model = smplx.create(**smplx_params).to(device)

    # Load data
    base_np = np.load(base_npz_path, allow_pickle=True)
    res_np = np.load(res_npz_path, allow_pickle=True)
    tar_np = np.load(tar_npz_path, allow_pickle=True)

    # Create folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get smplx faces
    faces = np.load(f"{smplx_params['model_path']}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    # Number of frames determined by middle (result)
    # n = res_np["poses"].shape[0]

    n = int(np.min([tar_np["poses"].shape[0], base_np["poses"].shape[0], res_np["poses"].shape[0]]))
    def npz_to_vertices(npz):
        beta = torch.from_numpy(npz["betas"]).to(torch.float32).unsqueeze(0).to(device)
        beta = beta.repeat(n, 1)
        expression = torch.from_numpy(npz["expressions"][:n]).to(torch.float32).to(device)
        jaw_pose = torch.from_numpy(npz["poses"][:n, 66:69]).to(torch.float32).to(device)
        pose = torch.from_numpy(npz["poses"][:n]).to(torch.float32).to(device)
        transl = torch.from_numpy(npz["trans"][:n]).to(torch.float32).to(device)
        output = smplx_model(
            betas=beta,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3 : 21 * 3 + 3],
            left_hand_pose=pose[:, 25 * 3 : 40 * 3],
            right_hand_pose=pose[:, 40 * 3 : 55 * 3],
            leye_pose=pose[:, 69:72],
            reye_pose=pose[:, 72:75],
            return_verts=True,
        )
        return output["vertices"].cpu().detach().numpy()

    # Convert all three sequences to vertices
    base_vertices = npz_to_vertices(base_np)
    res_vertices = npz_to_vertices(res_np)
    tar_vertices = npz_to_vertices(tar_np)

    # Silent video
    seconds = res_vertices.shape[0] // 30

    # Use three-panel path in fast_render by passing vertices_middle
    silent_video_file_path = generate_silent_videos(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        2,  # args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        base_vertices,  # left
        tar_vertices,  # right
        faces,
        output_dir,
        name,
        [],
        device,
        overlay=True,
        vertices_middle=res_vertices,
        titles=titles,
    )

    # Add audio
    if audio_path is not None:
        base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
        final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
    else:
        final_clip = silent_video_file_path

    return final_clip


def render_four_sequence(
    gt_npz_path,
    base_npz_path,
    res_npz_path,
    seg_npz_path,
    output_dir,
    args,
    smplx_params: dict = None,
    smplx_model=None,
    device="cpu",
    audio_path=None,
    name="",
    titles=None,
):
    """Render a four-panel video: GT (top-left), Base (top-right), RES (bottom-left), SEG (bottom-right).

    gt_npz_path:   path to NPZ for ground truth motion
    base_npz_path: path to NPZ for base model output (no integration)
    res_npz_path:  path to NPZ for final result (with integration)
    seg_npz_path:  path to NPZ for target semantic motion
    titles: optional list of four strings for [GT, Base, RES, SEG]
    """

    # Create smplx model
    if smplx_model is None:
        assert smplx_params is not None, "Define smplx parameters"
        smplx_model = smplx.create(**smplx_params).to(device)

    # Load data
    gt_np = np.load(gt_npz_path, allow_pickle=True)
    base_np = np.load(base_npz_path, allow_pickle=True)
    res_np = np.load(res_npz_path, allow_pickle=True)
    seg_np = np.load(seg_npz_path, allow_pickle=True)

    # Create folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get smplx faces
    faces = np.load(f"{smplx_params['model_path']}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    # Number of frames determined by result
    n = int(np.min([gt_np["poses"].shape[0], base_np["poses"].shape[0], res_np["poses"].shape[0], seg_np["poses"].shape[0]]))

    def npz_to_vertices(npz):
        beta = torch.from_numpy(npz["betas"]).to(torch.float32).unsqueeze(0).to(device)
        beta = beta.repeat(n, 1)
        expression = torch.from_numpy(npz["expressions"][:n]).to(torch.float32).to(device)
        jaw_pose = torch.from_numpy(npz["poses"][:n, 66:69]).to(torch.float32).to(device)
        pose = torch.from_numpy(npz["poses"][:n]).to(torch.float32).to(device)
        transl = torch.from_numpy(npz["trans"][:n]).to(torch.float32).to(device)
        output = smplx_model(
            betas=beta,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3 : 21 * 3 + 3],
            left_hand_pose=pose[:, 25 * 3 : 40 * 3],
            right_hand_pose=pose[:, 40 * 3 : 55 * 3],
            leye_pose=pose[:, 69:72],
            reye_pose=pose[:, 72:75],
            return_verts=True,
        )
        return output["vertices"].cpu().detach().numpy()

    # Convert all four sequences to vertices
    gt_vertices = npz_to_vertices(gt_np)
    base_vertices = npz_to_vertices(base_np)
    res_vertices = npz_to_vertices(res_np)
    seg_vertices = npz_to_vertices(seg_np)

    # Silent video
    seconds = res_vertices.shape[0] // 30

    # Use four-panel path in fast_render by passing all four vertex streams
    silent_video_file_path = generate_silent_videos(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        2,  # args.render_concurrent_num,
        args.render_tmp_img_filetype,
        int(seconds * args.render_video_fps),
        gt_vertices,  # top-left
        base_vertices,  # top-right
        faces,
        output_dir,
        name,
        [],
        0,
        overlay=True,
        vertices_middle=res_vertices,  # bottom-left
        vertices_fourth=seg_vertices,  # bottom-right
        titles=titles,
    )

    # Add audio
    if audio_path is not None:
        base_filename_without_ext = os.path.splitext(os.path.basename(res_npz_path))[0]
        final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
    else:
        final_clip = silent_video_file_path

    return final_clip


def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("CUDA version: {}".format(torch.version.cuda))
    logger.info("{} GPUs".format(torch.cuda.device_count()))
    logger.info(f"Random Seed: {args.random_seed}")


def args2csv(args, get_head=False, list4print=[]):
    for k, v in args.items():
        if isinstance(args[k], dict):
            args2csv(args[k], get_head, list4print)
        else:
            list4print.append(k) if get_head else list4print.append(v)
    return list4print


class EpochTracker:
    def __init__(self, metric_names, metric_directions):
        assert len(metric_names) == len(metric_directions), "Metric names and directions should have the same length"

        self.metric_names = metric_names
        self.states = ["train", "val", "test"]
        self.types = ["last", "best"]

        self.values = {
            name: {
                state: {type_: {"value": np.inf if not is_higher_better else -np.inf, "epoch": 0} for type_ in self.types} for state in self.states
            }
            for name, is_higher_better in zip(metric_names, metric_directions)
        }

        self.loss_meters = {name: {state: AverageMeter(f"{name}_{state}") for state in self.states} for name in metric_names}

        self.is_higher_better = {name: direction for name, direction in zip(metric_names, metric_directions)}
        self.train_history = {name: [] for name in metric_names}
        self.val_history = {name: [] for name in metric_names}

    def update_meter(self, name, state, value):
        self.loss_meters[name][state].update(value)

    def update_values(self, name, state, epoch):
        value_avg = self.loss_meters[name][state].avg
        new_best = False

        if (value_avg < self.values[name][state]["best"]["value"] and not self.is_higher_better[name]) or (
            value_avg > self.values[name][state]["best"]["value"] and self.is_higher_better[name]
        ):
            self.values[name][state]["best"]["value"] = value_avg
            self.values[name][state]["best"]["epoch"] = epoch
            new_best = True
        self.values[name][state]["last"]["value"] = value_avg
        self.values[name][state]["last"]["epoch"] = epoch
        return new_best

    def get(self, name, state, type_):
        return self.values[name][state][type_]

    def reset(self):
        for name in self.metric_names:
            for state in self.states:
                self.loss_meters[name][state].reset()

    def flatten_values(self):
        flat_dict = {}
        for name in self.metric_names:
            for state in self.states:
                for type_ in self.types:
                    value_key = f"{name}_{state}_{type_}"
                    epoch_key = f"{name}_{state}_{type_}_epoch"
                    flat_dict[value_key] = self.values[name][state][type_]["value"]
                    flat_dict[epoch_key] = self.values[name][state][type_]["epoch"]
        return flat_dict

    def update_and_plot(self, name, epoch, save_path):
        new_best_train = self.update_values(name, "train", epoch)
        new_best_val = self.update_values(name, "val", epoch)

        self.train_history[name].append(self.loss_meters[name]["train"].avg)
        self.val_history[name].append(self.loss_meters[name]["val"].avg)

        train_values = self.train_history[name]
        val_values = self.val_history[name]
        epochs = list(range(1, len(train_values) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_values, label="Train")
        plt.plot(epochs, val_values, label="Val")
        plt.title(f"Train vs Val {name} over epochs")
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

        return new_best_train, new_best_val


def record_trial(args, tracker):
    """
    1. record notes, score, env_name, experments_path,
    """
    csv_path = args.out_path + "custom/" + args.csv_name + ".csv"
    all_print_dict = vars(args)
    all_print_dict.update(tracker.flatten_values())
    if not os.path.exists(csv_path):
        pd.DataFrame([all_print_dict]).to_csv(csv_path, index=False)
    else:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame([all_print_dict])
        df_aligned = df_existing.append(df_new).fillna("")
        df_aligned.to_csv(csv_path, index=False)


def set_random_seed(args):
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic  # args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled


def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = {
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
            "opt_state": opt.state_dict(),
            "lrs": lrs.state_dict(),
        }
    elif opt is not None:
        states = {
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
            "opt_state": opt.state_dict(),
        }
    else:
        states = {
            "model_state": model.state_dict(),
        }
    torch.save(states, save_path)


def load_checkpoints(model, save_path, load_name="model"):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag = False
    for k, v in states["model_state"].items():
        if "module" not in k:
            break
        else:
            new_weights[k[7:]] = v
            flag = True
    if flag:
        try:
            model.load_state_dict(new_weights)
        except:
            model.load_state_dict(states["model_state"])
    else:
        model.load_state_dict(states["model_state"])
    logger.info(f"load self-pretrained checkpoints for {load_name}")


def model_complexity(model, args):
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), as_strings=False, print_per_layer_stat=False)
    logging.info("{:<30}  {:<8} BFlops".format("Computational complexity: ", flops / 1e9))
    logging.info("{:<30}  {:<8} MParams".format("Number of parameters: ", params / 1e6))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
