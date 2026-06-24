import os
import subprocess
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

def add_audio_to_video(silent_video_path, audio_path, output_video_path):
    command = [
        'ffmpeg',
        '-y',
        '-i', silent_video_path,
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        output_video_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Video with audio generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def convert_img_to_mp4(input_pattern, output_file, framerate=30):
    command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file,
        '-y' 
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Video conversion successful. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")


def add_subtitles(audio_video_file_path: str, subtitled_video_file_path: str, subtitles: list, text_frame_length: int = 196):
    output_srt_path = subtitled_video_file_path.replace(".mp4", ".srt")
    try:
        create_srt_file(subtitles, output_srt_path, 30, text_frame_length)
        if not add_subtitles_to_video(audio_video_file_path, output_srt_path, subtitled_video_file_path):
            return False
        return True
    except Exception as e:
        print(f"Error adding subtitles: {e}")
        return False


def add_subtitles_to_video(video_path, srt_path, output_video_path):
    """
    Adds subtitles to a video using ffmpeg.

    :param video_path: Path to the input video file.
    :param srt_path: Path to the subtitle (.srt) file.
    :param output_video_path: Path for the output video file with subtitles.
    """
    command = ["ffmpeg", "-y", "-i", video_path, "-vf", f"subtitles={srt_path}", output_video_path]

    try:
        subprocess.run(command, check=True)
        print(f"Video with subtitles generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    return True


def merge_two_videos_to_one(video_path1, video_path2, output_video_path, text1="", text2=""):
    """ """

    def ffmpeg_drawtext_escape(s: str) -> str:
        # Escape characters that drawtext treats specially
        return s.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")

    t0 = ffmpeg_drawtext_escape(text1)
    t1 = ffmpeg_drawtext_escape(text2)
    font = "DejaVu Sans"
    fc = (
        f"[0:v]scale=1280:-2,setsar=1,"  # ,pad=iw+180:ih:180:0:color=black
        f"drawtext=fontfile='{font}':text='{t0}':x=20:y=20:fontsize=32:fontcolor=black[v0];"
        f"[1:v]scale=1280:-2,setsar=1,"  # ,pad=iw+180:ih:180:0:color=black
        f"drawtext=fontfile='{font}':text='{t1}':x=20:y=20:fontsize=32:fontcolor=black[v1];"
        f"[v0][v1]vstack=inputs=2[v]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path1,
        "-i",
        video_path2,
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "0:a?",  # take audio from first if present (the '?' makes it optional)
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-shortest",
        output_video_path,
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Merged Video generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def create_srt_file(subtitles: list, output_srt_path: str, frame_rate: int = 30, text_length: int = 196):
    """
    Create an .srt file from a list of subtitles, with each title corresponding to a frame.

    :param subtitles: List of subtitle strings, one for each frame.
    :param output_srt_path: Output path for the .srt file.
    :param frame_rate: Frame rate of the video (default is 30).
    :param text_length: The text length in frames (default is 196).
    """
    text_time = text_length / frame_rate
    with open(output_srt_path, "w") as srt_file:
        for i, subtitle in enumerate(subtitles):
            # Calculate start and end times based on frame rate
            start_time_seconds = i * text_time
            end_time_seconds = (i + 1) * text_time

            # Convert seconds to SRT time format (HH:MM:SS,ms)
            start_time = "{:02}:{:02}:{:02},{:03}".format(
                int(start_time_seconds // 3600),
                int((start_time_seconds % 3600) // 60),
                int(start_time_seconds % 60),
                int((start_time_seconds % 1) * 1000),
            )

            end_time = "{:02}:{:02}:{:02},{:03}".format(
                int(end_time_seconds // 3600), int((end_time_seconds % 3600) // 60), int(end_time_seconds % 60), int((end_time_seconds % 1) * 1000)
            )

            # Write each subtitle block in SRT format
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{subtitle}\n\n")


def add_audio_to_video(silent_video_file_path: str, audio_video_file_path: str, audio_data: np.ndarray, audio_sr: int):
    """Save audio to file"""

    audio_path = audio_video_file_path.replace(".mp4", ".wav")
    sf.write(audio_path, audio_data, audio_sr)

    video = VideoFileClip(silent_video_file_path, audio=False)
    audio = AudioFileClip(audio_path)

    # Set the audio of the video clip
    video = video.set_audio(audio)

    # Save the final video with audio
    video.write_videofile(audio_video_file_path.replace("_silence", ""))

    video.close()

    os.remove(audio_path)
