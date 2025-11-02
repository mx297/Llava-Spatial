import os, re, numpy as np, shutil

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def extract_id(fname):
    # last integer group in the basename is treated as the frame id
    m = re.findall(r"(\d+)", os.path.splitext(os.path.basename(fname))[0])
    return int(m[-1]) if m else None

def sample_frame_ids_like_vlm3r(folder, num_frames, exts=(".jpg", ".jpeg", ".JPG", ".JPEG")):
    """
    Return selected FRAME IDS (ints) by reproducing VLM-3R's sampling:
      - candidates: all RGB images (assumed 1:1 with frames)
      - selection: np.linspace(0, N-1, num_frames, dtype=int) -> np.unique
    Assumes filenames contain the frame index (e.g., 000123.jpg or 123.jpg).
    """


    files = [f for f in os.listdir(folder) if f.endswith(exts)]
    if not files:
        return []

    files.sort(key=natural_key)
    #print(files)
    # Build parallel arrays of file->frame_id; require all IDs present
    # ids = []
    # keep = []
    # for f in files:
    #     fid = extract_id(f)
    #     if fid is not None:
    #         ids.append(fid)
    #         keep.append(f)
    # files = keep
    # if not files:
    #     return []

    N = len(files)

    # Exact same sampling logic as VLM-3R
    if N <= num_frames:
        idx = np.arange(N)
    else:
        idx = np.linspace(0, N - 1, num_frames, dtype=int)
        idx = np.unique(idx)  # may reduce count

    # Map to frame IDs (not file positions), preserving order
    #selected_ids = [extract_id(files[i]) for i in idx]
    return [os.path.join(folder, files[i]) for i in idx]


def main():
    folder = "scenes/scannet_2d"
    sampled_folder = "sampled_scenes"
    for scene in sorted(os.listdir(folder)):
        if '.txt' in scene:
            continue
        imagespath = os.path.join(folder,scene,'color')
        selected_files = sample_frame_ids_like_vlm3r(imagespath,32)
        print(f"f{len(selected_files)} frames are samples from scene {scene}")
        sampled_scene_path = os.path.join(sampled_folder,scene)
        if not os.path.exists(sampled_scene_path):
            os.mkdir(sampled_scene_path)
        
        sampled_frames_path = os.path.join(sampled_scene_path,'color')
        if not os.path.exists(sampled_frames_path):
            os.mkdir(sampled_frames_path)
        
        for file in selected_files:
            dst = os.path.join(sampled_frames_path,os.path.basename(file))
            shutil.copy2(file,dst)

if __name__ == '__main__':
    main()