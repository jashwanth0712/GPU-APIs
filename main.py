import cv2
import os
import tempfile
import subprocess
from pathlib import Path
from utils.caption_service import extract_captions, seconds_to_timecode, get_caption_for_time
from utils.watermark_service import process_frame
from utils.segmenting_service import process_scene, get_bounding_box
from faster_whisper import WhisperModel
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
#import nunmpy
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device ="cuda"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

def process_video(input_video_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        input_filename = Path(input_video_path).stem
        output_video_path = os.path.join(output_folder, f"{input_filename}_watermarked.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Warning: H264 codec failed, trying mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print("Extracting captions...")
        captions = extract_captions(input_video_path)
        print("Processing frames...")
        frame_count = 0
        bounding_box = None
        first_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count == 0:
                first_frame = frame.copy()
                bounding_box = get_bounding_box(first_frame)
                print(f"First frame bounding box: {bounding_box}")

            # Compute current timestamp in seconds
            current_time = frame_count / fps
            subtitle_text = get_caption_for_time(captions, current_time)

            # Add watermark and optionally caption
            processed_frame = process_frame(frame, subtitle_text)

            out.write(processed_frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()
        print(f"Processed {frame_count} frames")
        print(f"Output video saved: {output_video_path}")

        return bounding_box, output_video_path, captions

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None, None, None
    finally:
        print("Processing complete!")

def generate_foreground(video_path, bounding_box, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        input_filename = Path(video_path).stem
        output_video_path = os.path.join(output_folder, f"{input_filename}_segmented.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Warning: H264 codec failed, trying mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print("Processing frames for foreground segmentation...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Set the image for the predictor before predicting masks
            predictor.set_image(frame)
            point_coords = np.array([[960, 540]])  # center of frame
            point_labels = np.array([1])  # 1 for foreground, 0 for background
            # Apply segmentation using the bounding box
            input_box = np.array(bounding_box)
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box[None, :],
                multimask_output=False,
            )

            # Create a mask and apply it to the frame
            mask = masks[0]  # Assuming we want the first mask
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            segmented_frame = frame * cleaned_mask[:, :, np.newaxis]  # Apply mask to frame

            out.write(segmented_frame.astype(np.uint8))
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()
        print(f"Processed {frame_count} frames")
        print(f"Output segmented video saved: {output_video_path}")

        return output_video_path

    except Exception as e:
        print(f"Error generating foreground: {str(e)}")
        return None
    finally:
        print("Foreground generation complete!")

def main():
    input_path = input("Enter the path to your video file: ").strip().strip('"\'')
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist")
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    if not any(input_path.lower().endswith(ext) for ext in video_extensions):
        print("Warning: File doesn't have a common video extension")

    bounding_box, watermarked_video_path, captions = process_video(input_path)

    # Use detected bounding box if available, else fallback to full frame
    if bounding_box is None:
        print("No person detected in the first frame, using full frame as bounding box.")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bounding_box = [0, 0, width, height]
        cap.release()

    foreground_video_path = generate_foreground(input_path, bounding_box)

    # Now we have foreground, original video and captions 
    original_video_path = input_path
    # foreground video path is already set from generate_foreground function
    subtitle_text = captions if captions is not None else {}
    # The model of the subtitle is based on the extracted captions from the video, which are processed to align with the video frames.
    
    print(f"\nProcessing Summary:")
    print(f"Original video: {original_video_path}")
    print(f"Watermarked video: {watermarked_video_path}")
    print(f"Foreground video: {foreground_video_path}")
    print(f"Captions extracted: {len(subtitle_text) if subtitle_text else 0} segments")


if __name__ == "__main__":
    main()
