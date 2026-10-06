import cv2
import os
import time
import argparse
from PIL import Image
import imagehash

def extract_frames(video_path: str, 
                   output_dir: str,
                   interval_sec: float = 1.0,
                   hash_threshold: int = 5):
    
    """Extract frames from a video file at a set interval, using perceptual hashing to drop near-duplicate frames"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return

    os.makedirs(output_dir, exist_ok = True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Error: Could not read video FPS")
        return

    frame_interval = int(fps * interval_sec)

    frame_count = 0
    saved_count = 0
    last_hash = None

    print(f"Processing '{video_path}' at {fps} FPS...")
    print(f"Target interval: {interval_sec}s (every {frame_interval} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BG2BGR))
            current_hash = imagehash.phash(pil_img)

            if last_hash is None or (current_hash - last_hash) > hash_threshold:
                timestamp = int(time.time())
                filename = os.path.join(output_dir, f"dashcam_{timestamp}_{saved_count:04d}.jpg")

                cv2.imwrite(filename, frame)
                last_hash = current_hash
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete! Saved {saved_count} unique frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract distinct frames from dashcam footage.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video (.mp4, .avi)")
    parser.add_argument("--out", type=str, default="../data/raw", help="Output directory for JPEGs")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between extracted frames")
    
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.interval)