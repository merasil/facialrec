import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fmt='png'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.{fmt}")
        cv2.imwrite(filename, frame)
        frame_idx += 1
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract every frame from a video")
    parser.add_argument("video", help="Pfad zur Videodatei")
    parser.add_argument("output", help="Ausgabeordner für Frames")
    parser.add_argument("--format", choices=['png','jpg'], default='png',
                        help="Bildformat: png (verlustfrei) oder jpg")
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.format)
