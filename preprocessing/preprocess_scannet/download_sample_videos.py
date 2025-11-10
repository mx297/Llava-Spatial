import argparse
import os
import sys
import cv2  # Requires opencv-python
import numpy as np
from tqdm import tqdm  # Requires tqdm
import concurrent.futures  # Parallel processing
import subprocess  # NEW: for downloader
import shutil      # NEW: for deleting downloaded scenes
import pandas as pd  # NEW: to read JSONL
from sensor import SensorData

def get_scannet_scene_ids_from_jsonl(jsonl_path):
    """Return unique ScanNet scene IDs from a JSONL with 'data_source' and 'scene_name'."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return []
    if 'data_source' not in df.columns or 'scene_name' not in df.columns:
        print(f"Error: {jsonl_path} must contain 'data_source' and 'scene_name' columns.")
        return []
    return sorted(np.unique(df[df['data_source'] == 'scannet']['scene_name']).tolist())

def download_scannet_scene(scannet_script, out_dir, scene_id):
    """Calls scannet.py to download a single scene into out_dir. Returns True on success."""
    cmd = [sys.executable, scannet_script, "-o", out_dir, "--id", scene_id]
    input_data = "\n\n"  # accept both prompts
    try:
        subprocess.run(cmd, input=input_data, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed for scene {scene_id}: {e}")
        return False
        
def export_scene_video(sens_file_path, output_video_path, width, height, fps, frame_skip, codec):
    """Exports video for a single scene's .sens file."""
    print(f"Processing scene: {os.path.basename(os.path.dirname(sens_file_path))}")
    print(f'Loading {sens_file_path}...')
    try:
        sd = SensorData(sens_file_path)
    except NameError:
        print("Error: SensorData class failed to be imported.")
        return False
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return False
    print(f'Loaded {len(sd.frames)} frames.')

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData object does not contain frames or is empty.")
        return False
    if not hasattr(sd, 'color_compression_type'):
        print("Error: SensorData object does not have 'color_compression_type' attribute.")
        return False
    if not hasattr(sd.frames[0], 'decompress_color'):
        print("Error: Frame objects in SensorData do not have 'decompress_color'.")
        return False

    image_size = (width, height)  # (width, height) for OpenCV
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

    if not writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}.")
        return False

    print(f"Exporting video to {output_video_path} with resolution {image_size} and FPS {fps}...")

    frame_indices = range(0, len(sd.frames), frame_skip)
    processed_frames = 0
    skipped_frames = 0
    success = True

    try:
        for i in tqdm(frame_indices, desc="Processing frames", leave=False):
            try:
                frame = sd.frames[i]
                color_image_rgb = frame.decompress_color(sd.color_compression_type)
            except IndexError:
                print(f"Warning: Frame index {i} out of bounds. Stopping scene processing.")
                success = False
                break
            except Exception as e:
                print(f"Warning: Could not decompress color for frame {i}: {e}. Skipping.")
                skipped_frames += 1
                continue

            if color_image_rgb is None:
                print(f"Warning: Null color image for frame {i}. Skipping.")
                skipped_frames += 1
                continue

            if not isinstance(color_image_rgb, np.ndarray):
                try:
                    color_image_rgb = np.array(color_image_rgb)
                except Exception as e_conv:
                    print(f"Warning: Conversion to ndarray failed for frame {i}: {e_conv}. Skipping.")
                    skipped_frames += 1
                    continue

            if len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 3:
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            elif len(color_image_rgb.shape) == 2:
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_GRAY2BGR)
            elif len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 4:
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGBA2BGR)
            else:
                print(f"Warning: Unexpected image shape {color_image_rgb.shape} at frame {i}. Skipping.")
                skipped_frames += 1
                continue

            current_height, current_width = color_image_bgr.shape[:2]
            if current_width != width or current_height != height:
                resized_image = cv2.resize(color_image_bgr, image_size, interpolation=cv2.INTER_AREA)
            else:
                resized_image = color_image_bgr

            writer.write(resized_image)
            processed_frames += 1

    except Exception as e:
        print(f"Unexpected error during processing for {sens_file_path}: {e}")
        success = False
    finally:
        if writer.isOpened():
            writer.release()

    if success:
        print(f'Finished exporting video for scene.')
        print(f'Processed {processed_frames} frames.')
        if skipped_frames > 0:
            print(f'Skipped {skipped_frames} frames due to errors.')
        print(f'Video saved to: {output_video_path}')
    else:
        print(f"Failed to fully export video for scene {os.path.basename(os.path.dirname(sens_file_path))}.")
        if os.path.exists(output_video_path) and processed_frames == 0:
            print(f"Deleting incomplete file: {output_video_path}")
            try:
                os.remove(output_video_path)
            except OSError as e_del:
                print(f"Error deleting file {output_video_path}: {e_del}")

    return success


def worker_download_process_delete(
    scene_id,
    base_download_dir,
    download_dir,
    output_dir,
    width,
    height,
    fps,
    frame_skip,
    codec,
    train_scenes,
    val_scenes,
    use_splits,
    scannet_script,
):
    """Full lifecycle for one scene. Returns (scene_id, split, success, message)."""
    # 1) Download
    ok = download_scannet_scene(scannet_script, base_download_dir, scene_id)
    if not ok:
        return (scene_id, "unknown", False, "download_failed")

    scene_dir_path = os.path.join(download_dir, scene_id)
    sens_file_path = os.path.join(scene_dir_path, f"{scene_id}.sens")

    # Pick subdir for output video (train/val or base)
    split = ""
    if use_splits:
        if scene_id in train_scenes:
            split = "train"
        elif scene_id in val_scenes:
            split = "val"
        # else leave as ""

    # Determine extension from codec
    ext = ".mp4"
    lc = codec.lower()
    if lc in ['mjpg']:
        ext = ".avi"

    output_video_dir = os.path.join(output_dir, split) if split else output_dir
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{scene_id}{ext}")

    try:
        if not os.path.exists(sens_file_path):
            return (scene_id, split or "nosplit", False, "sens_missing_after_download")

        # Skip if output already exists
        if os.path.exists(output_video_path):
            return (scene_id, split or "nosplit", True, "already_exists")

        ok = export_scene_video(
            sens_file_path=sens_file_path,
            output_video_path=output_video_path,
            width=width, height=height,
            fps=fps, frame_skip=frame_skip, codec=codec
        )
        return (scene_id, split or "nosplit", bool(ok), "ok" if ok else "processing_failed")

    except Exception as e:
        return (scene_id, split or "nosplit", False, f"processing_exception:{e}")

    finally:
        # 3) Delete local scene to free storage
        try:
            shutil.rmtree(scene_dir_path, ignore_errors=True)
        except Exception:
            pass

# ---------------------------
# CLI & Orchestration
# ---------------------------

parser = argparse.ArgumentParser(description="Download ScanNet scenes, export color video from .sens, then delete scenes.")
parser.add_argument('--output_dir', required=True, help='Directory where output videos will be saved.')
parser.add_argument('--train_val_splits_path', required=False, help='Directory containing scannetv2_train.txt and scannetv2_val.txt.')
parser.add_argument('--jsonl_path', required=True, help='Path to JSONL (e.g., vlm3r.jsonl) listing scene IDs.')
parser.add_argument('--scannet_script', type=str, default='scannet.py', help='Path to scannet.py downloader.')
# video settings
parser.add_argument('--width', type=int, required=True, help='output video width')
parser.add_argument('--height', type=int, required=True, help='output video height')
parser.add_argument('--fps', type=int, default=30, help='output video frame rate (default: 30)')
parser.add_argument('--frame_skip', type=int, default=1, help='process every nth frame (default: 1)')
parser.add_argument('--codec', type=str, default='mp4v', help='video codec (default: mp4v for .mp4). Use "avc1" for H.264.')
# parallelism cap == max scenes on disk at once
parser.add_argument('--max_inflight_scenes', type=int, default=5,
                    help='Max number of scenes simultaneously present in --scans_dir. Defaults to CPU count.')


def main():

    opt = parser.parse_args()
    print("Script Options:")
    print(opt)
    BASE_DIR = "/l/users/mohamed.abouelhadid/videos/"
    DOWNLOAD_DIR = BASE_DIR + 'scans'

    # Load splits if provided
    train_scenes = set()
    val_scenes = set()
    use_splits = False
    if opt.train_val_splits_path:
        train_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_train.txt')
        val_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_val.txt')
        try:
            with open(train_file_path) as f:
                train_scenes = set(f.read().splitlines())
            with open(val_file_path) as f:
                val_scenes = set(f.read().splitlines())
            use_splits = True
            print(f"Loaded {len(train_scenes)} train and {len(val_scenes)} val scenes.")
        except Exception as e:
            print(f"Warning: could not read split files: {e}. Proceeding without split subfolders.")

    os.makedirs(opt.output_dir, exist_ok=True)
    if use_splits:
        os.makedirs(os.path.join(opt.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)

    # Build scene queue from JSONL
    #scene_ids = get_scannet_scene_ids_from_jsonl(opt.jsonl_path)
    scene_ids = ['scene0585_00']
    if not scene_ids:
        print("No scene IDs found. Exiting.")
        sys.exit(0)
    print(f"Found {len(scene_ids)} scene IDs in {opt.jsonl_path}.")

    processed_ok = 0
    processed_fail = 0

    max_workers = opt.max_inflight_scenes
    print(f"Starting parallel Download→Export→Delete with up to {max_workers} scenes in flight...")

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for sid in scene_ids:
            fut = ex.submit(
                worker_download_process_delete,
                sid,
                BASE_DIR,
                DOWNLOAD_DIR,
                opt.output_dir,
                opt.width, opt.height, opt.fps, opt.frame_skip, opt.codec,
                train_scenes, val_scenes, use_splits,
                opt.scannet_script
            )
            futures.append(fut)

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Download→Export→Delete"):
            try:
                sid, split, ok, msg = fut.result()
                if ok:
                    processed_ok += 1
                else:
                    processed_fail += 1
                print(f"[{split}] {sid}: {msg}")
            except Exception as e:
                processed_fail += 1
                print(f"Worker retrieval exception: {e}")

    print("\n" + "=" * 30)
    print("Processing Summary:")
    print(f"Scenes attempted: {len(scene_ids)}")
    print(f"Successfully exported: {processed_ok}")
    print(f"Failed: {processed_fail}")  
    print("=" * 30)

if __name__ == '__main__':
    main()
