#!/usr/bin/env python3
"""
Crop a specific region from a grid video.
Example: Extract top-right 3x3 grid from a 5x5 grid video.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def crop_grid_video(
    input_video: str,
    output_video: str,
    original_grid: tuple = (5, 5),
    crop_position: str = "top-right",
    crop_size: tuple = (3, 3)
):
    """
    Crop a specific region from a grid video.
    
    Args:
        input_video: Path to input grid video
        output_video: Path to output cropped video
        original_grid: Original grid size as (rows, cols)
        crop_position: Position of crop region:
            - "top-left", "top-right", "bottom-left", "bottom-right"
            - "center"
            - Or custom coordinates as "row,col" (e.g., "0,2" for row 0, col 2)
        crop_size: Crop size as (rows, cols)
    
    Returns:
        True if successful
    """
    print(f"{'='*60}")
    print(f"Cropping Grid Video")
    print(f"{'='*60}")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Original grid: {original_grid[0]}x{original_grid[1]}")
    print(f"Crop region: {crop_size[0]}x{crop_size[1]} from {crop_position}")
    print(f"{'='*60}\n")
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"❌ Failed to open input video: {input_video}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Input video info:")
    print(f"  Resolution: {orig_width}x{orig_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Calculate cell dimensions
    orig_rows, orig_cols = original_grid
    cell_width = orig_width // orig_cols
    cell_height = orig_height // orig_rows
    
    print(f"📐 Grid cell size: {cell_width}x{cell_height}")
    print()
    
    # Determine crop region starting position (row, col)
    crop_rows, crop_cols = crop_size
    
    if crop_position == "top-left":
        start_row, start_col = 0, 0
    elif crop_position == "top-right":
        start_row, start_col = 0, orig_cols - crop_cols
    elif crop_position == "bottom-left":
        start_row, start_col = orig_rows - crop_rows, 0
    elif crop_position == "bottom-right":
        start_row, start_col = orig_rows - crop_rows, orig_cols - crop_cols
    elif crop_position == "center":
        start_row = (orig_rows - crop_rows) // 2
        start_col = (orig_cols - crop_cols) // 2
    elif "," in crop_position:
        # Custom position as "row,col"
        start_row, start_col = map(int, crop_position.split(','))
    else:
        print(f"❌ Invalid crop_position: {crop_position}")
        return False
    
    # Validate crop region
    if start_row < 0 or start_col < 0 or \
       start_row + crop_rows > orig_rows or \
       start_col + crop_cols > orig_cols:
        print(f"❌ Invalid crop region: ({start_row}, {start_col}) + ({crop_rows}, {crop_cols})")
        print(f"   Exceeds original grid: {orig_rows}x{orig_cols}")
        return False
    
    # Calculate pixel coordinates
    x1 = start_col * cell_width
    y1 = start_row * cell_height
    x2 = x1 + crop_cols * cell_width
    y2 = y1 + crop_rows * cell_height
    
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    print(f"✂️ Crop region:")
    print(f"  Grid position: row {start_row}-{start_row + crop_rows - 1}, col {start_col}-{start_col + crop_cols - 1}")
    print(f"  Pixel coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Output resolution: {crop_width}x{crop_height}")
    print()
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (crop_width, crop_height))
    
    if not out.isOpened():
        print(f"❌ Failed to create output video writer")
        cap.release()
        return False
    
    # Process frames
    print(f"⏳ Processing frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        cropped_frame = frame[y1:y2, x1:x2]
        
        # Write to output
        out.write(cropped_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Progress: {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get output file size
    output_path = Path(output_video)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Cropping completed!")
    print(f"📹 Output: {output_video}")
    print(f"📦 File size: {file_size_mb:.2f} MB")
    print(f"🎬 Processed {frame_count} frames")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Crop specific region from grid video")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input grid video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output cropped video file")
    parser.add_argument("--original_grid", type=str, default="5x5", help="Original grid layout (e.g., '5x5')")
    parser.add_argument("--crop_size", type=str, default="3x3", help="Crop size (e.g., '3x3')")
    parser.add_argument("--position", "-p", type=str, default="top-right", 
                       help="Crop position: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', or 'row,col'")
    
    args = parser.parse_args()
    
    # Parse grid sizes
    orig_rows, orig_cols = map(int, args.original_grid.split('x'))
    crop_rows, crop_cols = map(int, args.crop_size.split('x'))
    
    # Perform cropping
    success = crop_grid_video(
        input_video=args.input,
        output_video=args.output,
        original_grid=(orig_rows, orig_cols),
        crop_position=args.position,
        crop_size=(crop_rows, crop_cols)
    )
    
    if success:
        print("\n✅ Success!")
    else:
        print("\n❌ Failed!")
        exit(1)


if __name__ == "__main__":
    main()
