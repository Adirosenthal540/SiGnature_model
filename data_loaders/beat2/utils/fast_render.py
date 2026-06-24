import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import time
import numpy as np
import pyrender
import torch
import trimesh
import queue
import imageio
import threading
import multiprocessing
from data_loaders.beat2.utils.media import convert_img_to_mp4
import glob
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import os
from multiprocessing import get_context

def deg_to_rad(degrees):
    return degrees * np.pi / 180


def create_pose_camera(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
            [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def create_pose_light(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light, overlay=False):
    if overlay:
        scene = pyrender.Scene()
        for i, v in enumerate(vertices):
            trimesh_mesh = trimesh.Trimesh(vertices=v, faces=faces, vertex_colors=uniform_color[i])
            mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
            scene.add(mesh)
    else:
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=uniform_color)
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
        scene = pyrender.Scene()
        scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    return scene


def create_overlay_scene(vertices_result, vertices_benchmark, faces, pose_camera, pose_light):
    """Create a scene with both models overlaid with transparency"""
    import pyrender
    import trimesh

    scene = pyrender.Scene()

    # Create meshes for both models
    # Result model (semi-transparent blue)
    uniform_color_result = [100, 150, 255, 255]  # Semi-transparent blue
    trimesh_mesh_result = trimesh.Trimesh(vertices=vertices_result, faces=faces, vertex_colors=uniform_color_result)
    mesh_result = pyrender.Mesh.from_trimesh(trimesh_mesh_result, smooth=True)
    scene.add(mesh_result)

    # Benchmark model (semi-transparent red)
    uniform_color_benchmark = [255, 100, 100, 255]  # Semi-transparent red
    trimesh_mesh_benchmark = trimesh.Trimesh(vertices=vertices_benchmark, faces=faces, vertex_colors=uniform_color_benchmark)
    mesh_benchmark = pyrender.Mesh.from_trimesh(trimesh_mesh_benchmark, smooth=True)
    scene.add(mesh_benchmark)

    # Add camera and lighting
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)

    return scene


def do_render_one_frame(renderer, frame_idx, vertices, faces, is_keyframe=None, overlay=False):
    if frame_idx % 100 == 0:
        print("processed", frame_idx, "frames")

    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)

    figs = []

    if overlay:
        # Handle different panel configurations
        if len(vertices) == 4:
            # Four-panel mode: GT (top-left), Base (top-right), RES (bottom-left), SEG (bottom-right)
            colors = [
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),  # GT
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),  # Base
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),  # RES
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),  # SEG
            ]
            for idx in range(4):
                scene_i = create_scene_with_mesh(vertices[idx], faces, colors[idx], pose_camera, pose_light, overlay=False)
                fig_i, _ = renderer.render(scene_i)
                figs.append(fig_i)
        elif len(vertices) == 3:
            # Three-panel mode: Base (left), RES (middle), Target (right)
            colors = [
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),
                ([220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]),
            ]
            for idx in range(3):
                scene_i = create_scene_with_mesh(vertices[idx], faces, colors[idx], pose_camera, pose_light, overlay=False)
                fig_i, _ = renderer.render(scene_i)
                figs.append(fig_i)
        else:
            # Legacy: 2 streams with an overlay in the middle
            # Left frame: Result only
            uniform_color_result = [220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]
            scene_result = create_scene_with_mesh(vertices[0], faces, uniform_color_result, pose_camera, pose_light, overlay=False)
            fig_result, _ = renderer.render(scene_result)
            figs.append(fig_result)

            # Middle frame: Overlay of both models with transparency
            scene_overlay = create_overlay_scene(vertices[0], vertices[1], faces, pose_camera, pose_light)
            fig_overlay, _ = renderer.render(scene_overlay)
            figs.append(fig_overlay)

            # Right frame: Benchmark only
            uniform_color_benchmark = [220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]
            scene_benchmark = create_scene_with_mesh(vertices[1], faces, uniform_color_benchmark, pose_camera, pose_light, overlay=False)
            fig_benchmark, _ = renderer.render(scene_benchmark)
            figs.append(fig_benchmark)

    else:
        # Regular side-by-side mode
        uniform_color = [220, 220, 220, 255] if not is_keyframe else [255, 0, 0, 255]
        for vtx in vertices:
            scene = create_scene_with_mesh(vtx, faces, uniform_color, pose_camera, pose_light, overlay)
            fig, _ = renderer.render(scene)
            figs.append(fig)

    return figs


def do_render_one_frame_no_gt(renderer, frame_idx, vertices, faces):
    if frame_idx % 100 == 0:
        print("processed", frame_idx, "frames")

    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)

    figs = []
    # for vtx in [vertices]:
    scene = create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light)
    fig, _ = renderer.render(scene)
    figs.append(fig)

    return figs[0]


def add_title_to_image(fig, title):
    """Add a title to the top of an image"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    # Convert the figure (numpy array) to a PIL image for easy manipulation
    img = Image.fromarray(fig)

    # Initialize drawing context
    draw = ImageDraw.Draw(img)

    # Try to load a larger font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

    # Add title text at the top of the image
    text_position = (10, 10)  # Position the text near the top left
    draw.text(text_position, title, font=font, fill=(0, 0, 0))  # White text

    return np.array(img)  # Convert back to numpy array


def add_subtitle_to_image(fig, subtitle):
    # Convert the figure (numpy array) to a PIL image for easy manipulation
    img = Image.fromarray(fig)

    # Initialize drawing context
    draw = ImageDraw.Draw(img)

    # Optional: You can load a custom font, otherwise it uses a default font
    # font = ImageFont.truetype("arial.ttf", 24)  # Load a TTF font file
    font = ImageFont.load_default()

    # Add subtitle text at the bottom of the image
    text_position = (10, img.size[1] - 30)  # Position the text near the bottom left
    draw.text(text_position, subtitle, font=font, fill=(255, 255, 255))  # White text

    return np.array(img)  # Convert back to numpy array


def write_images_from_queue(fig_queue, output_dir, img_filetype, overlay=False, titles=None):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        if not overlay:
            fid, fig1, fig2 = e
            filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")

            # Add titles to each side if provided
            if titles is not None and len(titles) >= 2:
                fig1_with_title = add_title_to_image(fig1, titles[0])
                fig2_with_title = add_title_to_image(fig2, titles[1])
                merged_fig = np.hstack((fig1_with_title, fig2_with_title))
            else:
                merged_fig = np.hstack((fig1, fig2))
        else:
            # Handle different numbers of figures for overlay mode
            if len(e) == 5:  # Four-panel mode: fid, fig1, fig2, fig3, fig4
                fid, fig1, fig2, fig3, fig4 = e
                filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")

                # Add titles to each panel if provided
                if titles is not None and len(titles) >= 4:
                    fig1_with_title = add_title_to_image(fig1, titles[0])  # GT
                    fig2_with_title = add_title_to_image(fig2, titles[1])  # Base
                    fig3_with_title = add_title_to_image(fig3, titles[2])  # RES
                    fig4_with_title = add_title_to_image(fig4, titles[3])  # SEG

                    # Create single row layout: GT, Base, RES, SEG
                    merged_fig = np.hstack((fig1_with_title, fig2_with_title, fig3_with_title, fig4_with_title))
                else:
                    # Create single row layout without titles
                    merged_fig = np.hstack((fig1, fig2, fig3, fig4))
            else:  # Three-panel mode: fid, fig1, fig2, fig3
                fid, fig1, fig2, fig3 = e
                filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")

                # Add titles to each side if provided (for overlay mode)
                if titles is not None and len(titles) >= 3:
                    fig1_with_title = add_title_to_image(fig1, titles[0])
                    fig2_with_title = add_title_to_image(fig2, titles[1])
                    fig3_with_title = add_title_to_image(fig3, titles[2])
                    merged_fig = np.hstack((fig1_with_title, fig2_with_title, fig3_with_title))
                else:
                    merged_fig = np.hstack((fig1, fig2, fig3))

        try:
            imageio.imwrite(filename, merged_fig)
        except Exception as ex:
            print(f"Error writing image {filename}: {ex}")
            raise ex


def write_images_from_queue_no_gt(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1, fig2 = e
        filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        merged_fig = fig1  # np.hstack((fig1))
        try:
            imageio.imwrite(filename, merged_fig)
        except Exception as ex:
            print(f"Error writing image {filename}: {ex}")
            raise ex


def render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue, is_keyframe=None, overlay=False):
    # Adjust resolution based on number of panels
    verts_tuple = frame_vertex_pairs[0] if frame_vertex_pairs else None
    verts_list = list(verts_tuple) if isinstance(verts_tuple, tuple) else verts_tuple
    num_panels = len(verts_list) if overlay and verts_list else (3 if overlay else 2)

    if num_panels == 4:
        # Four-panel: 2x2 layout, each panel is half the width and height
        fig_resolution = (render_width // 4, render_height)
    elif num_panels == 3:
        # Three-panel: horizontal layout
        fig_resolution = (render_width // 3, render_height)
    else:
        # Two-panel: horizontal layout
        fig_resolution = (render_width // 2, render_height)

    renderer = pyrender.OffscreenRenderer(*fig_resolution)

    for idx, fid in enumerate(fids):
        if not overlay:
            fig1, fig2 = do_render_one_frame(
                renderer,
                fid,
                [frame_vertex_pairs[idx][0], frame_vertex_pairs[idx][1]],
                faces,
                is_keyframe[idx] if is_keyframe is not None else None,
                overlay=False,
            )
            fig_queue.put((fid, fig1, fig2))
        else:
            # Overlay mode: handle different numbers of streams
            verts_tuple = frame_vertex_pairs[idx]
            verts_list = list(verts_tuple) if isinstance(verts_tuple, tuple) else verts_tuple

            if len(verts_list) == 4:
                # Four-panel mode: GT, Base, RES, SEG
                fig1, fig2, fig3, fig4 = do_render_one_frame(
                    renderer,
                    fid,
                    [verts_list[0], verts_list[1], verts_list[2], verts_list[3]],
                    faces,
                    is_keyframe[idx] if is_keyframe is not None else None,
                    overlay=True,
                )
                fig_queue.put((fid, fig1, fig2, fig3, fig4))
            elif len(verts_list) == 3:
                # Three-panel mode: Base, RES, Target
                fig1, fig2, fig3 = do_render_one_frame(
                    renderer,
                    fid,
                    [verts_list[0], verts_list[1], verts_list[2]],
                    faces,
                    is_keyframe[idx] if is_keyframe is not None else None,
                    overlay=True,
                )
                fig_queue.put((fid, fig1, fig2, fig3))
            else:
                # Legacy 2-stream overlay
                fig1, fig2, fig3 = do_render_one_frame(
                    renderer,
                    fid,
                    [verts_list[0], verts_list[1]],  # legacy 2-stream overlay
                    faces,
                    is_keyframe[idx] if is_keyframe is not None else None,
                    overlay=True,
                )
                fig_queue.put((fid, fig1, fig2, fig3))

    renderer.delete()


def render_frames_and_enqueue_no_gt(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width // 2, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)

    for idx, fid in enumerate(fids):
        fig1 = do_render_one_frame_no_gt(renderer, fid, frame_vertex_pairs[idx][0], faces)
        fig_queue.put((fid, fig1))

    renderer.delete()


def sub_process_process_frame(
    subprocess_index,
    render_video_width,
    render_video_height,
    render_tmp_img_filetype,
    fids,
    frame_vertex_pairs,
    faces,
    output_dir,
    is_keyframe=None,
    device_id=0,
    overlay=False,
    titles=None,
):
    torch.cuda.set_device(device_id)
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue, is_keyframe, overlay)
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(
        target=write_images_from_queue,
        args=(fig_queue, output_dir, render_tmp_img_filetype, overlay, titles),
    )
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )


def sub_process_process_frame_no_gt(
    subprocess_index,
    render_video_width,
    render_video_height,
    render_tmp_img_filetype,
    fids,
    frame_vertex_pairs,
    faces,
    output_dir,
):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(
        fids,
        frame_vertex_pairs,
        faces,
        render_video_width,
        render_video_height,
        fig_queue,
    )
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(
        target=write_images_from_queue_no_gt,
        args=(fig_queue, output_dir, render_tmp_img_filetype),
    )
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )


def distribute_frames(
    frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all, overlay=False, vertices_middle=None, vertices_fourth=None
):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)

        if overlay:
            # In overlay mode, support optional third and fourth streams
            if vertices_fourth is not None and vertices_middle is not None:
                # Four-panel mode: GT, Base, RES, SEG
                subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i], vertices_middle[i], vertices_fourth[i]))
            elif vertices_middle is not None:
                # Three-panel mode: Base, RES, Target
                subproc_vertices[subprocess_index].append((vertices_all[i], vertices_middle[i], vertices1_all[i]))
            else:
                # Two-panel mode: Base, Target
                subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i]))
        else:
            subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i]))

        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices


def distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append((vertices_all[i], vertices_all[i]))
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices


def generate_silent_videos(
    render_video_fps,
    render_video_width,
    render_video_height,
    render_concurent_nums,
    render_tmp_img_filetype,
    frames,
    vertices_all,
    vertices1_all,
    faces,
    output_dir,
    name="",
    keyframes=[],
    device_id=0,
    overlay=False,
    vertices_middle=None,
    vertices_fourth=None,
    titles=None,
):

    subproc_frame_ids, subproc_vertices = distribute_frames(
        frames,
        render_video_fps,
        render_concurent_nums,
        vertices_all,
        vertices1_all,
        overlay=overlay,
        vertices_middle=vertices_middle,
        vertices_fourth=vertices_fourth,
    )

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    # ctx = get_context("spawn")
    if len(keyframes) > 0:
        keyframes = [np.array(keyframes)[subproc_frame_ids[subprocess_index]] for subprocess_index in range(render_concurent_nums)]
    else:
        keyframes = [None for subprocess_index in range(render_concurent_nums)]

    for subprocess_index in range(render_concurent_nums):
        sub_process_process_frame(
            subprocess_index,
            render_video_width,
            render_video_height,
            render_tmp_img_filetype,
            subproc_frame_ids[subprocess_index],
            subproc_vertices[subprocess_index],
            faces,
            output_dir,
            keyframes[subprocess_index],
            device_id,
            overlay,
            titles,
        )
    # with ctx.Pool(render_concurent_nums) as pool:
    #     pool.starmap(
    #         sub_process_process_frame,
    #         [
    #             (
    #                 subprocess_index,
    #                 render_video_width,
    #                 render_video_height,
    #                 render_tmp_img_filetype,
    #                 subproc_frame_ids[subprocess_index],
    #                 subproc_vertices[subprocess_index],
    #                 faces,
    #                 output_dir,
    #                 keyframes[subprocess_index],
    #                 device_id,
    #                 overlay,
    #             )
    #             for subprocess_index in range(render_concurent_nums)
    #         ],
    #     )

    output_file = os.path.join(output_dir, f"{name}_silence_video.mp4")
    convert_img_to_mp4(
        os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"),
        output_file,
        render_video_fps,
    )
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file


def generate_silent_videos_no_gt(
    render_video_fps,
    render_video_width,
    render_video_height,
    render_concurent_nums,
    render_tmp_img_filetype,
    frames,
    vertices_all,
    faces,
    output_dir,
):

    subproc_frame_ids, subproc_vertices = distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame_no_gt,
            [
                (
                    subprocess_index,
                    render_video_width,
                    render_video_height,
                    render_tmp_img_filetype,
                    subproc_frame_ids[subprocess_index],
                    subproc_vertices[subprocess_index],
                    faces,
                    output_dir,
                )
                for subprocess_index in range(render_concurent_nums)
            ],
        )

    output_file = os.path.join(output_dir, "silence_video.mp4")
    convert_img_to_mp4(
        os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"),
        output_file,
        render_video_fps,
    )
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file
