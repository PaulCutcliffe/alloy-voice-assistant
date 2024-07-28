import os
import glob
from PIL import Image
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(script_name):
    now = datetime.now()
    current_date_formatted = now.strftime('%Y-%m-%d')
    current_time_formatted = now.strftime('%H-%M-%S')

    if not os.path.exists('LogFiles'):
        os.makedirs('LogFiles')

    log_filename = f'LogFiles/{script_name}-{current_date_formatted}-{current_time_formatted}.log'

    logger = logging.getLogger('gif_stitcher')
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler(log_filename, maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def stitch_gifs(input_folder, output_file, remove_pause_frames=False, pause_frame_count=4):
    logger = setup_logger('gif_stitcher')
    logger.info(f"Starting GIF stitching process. Input folder: {input_folder}, Output file: {output_file}")

    gif_files = sorted(glob.glob(os.path.join(input_folder, '*.gif')))
    if not gif_files:
        logger.error(f"No GIF files found in {input_folder}")
        return

    first_gif = Image.open(gif_files[0])
    frame_width, frame_height = first_gif.size
    fps = 10  # You can adjust this value

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    total_frames = 0

    for gif_file in gif_files:
        logger.info(f"Processing GIF: {gif_file}")
        gif = Image.open(gif_file)
        frame_count = 0

        for frame in range(0, gif.n_frames):
            gif.seek(frame)
            frame_image = gif.convert("RGB")
            
            if remove_pause_frames and frame >= gif.n_frames - pause_frame_count:
                logger.debug(f"Skipping pause frame {frame} in {gif_file}")
                continue

            frame_array = np.array(frame_image)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_array)
            frame_count += 1

        logger.info(f"Added {frame_count} frames from {gif_file}")
        total_frames += frame_count

    video_writer.release()
    logger.info(f"Video creation complete. Total frames: {total_frames}")
    logger.info(f"Output video saved as: {output_file}")

if __name__ == "__main__":
    input_folder = "interactions"  # Change this to your GIF folder path
    output_file = "stitched_video.mp4"
    remove_pause_frames = True  # Set to False if you want to keep pause frames

    stitch_gifs(input_folder, output_file, remove_pause_frames)