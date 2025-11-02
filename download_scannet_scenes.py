import pandas as pd
import numpy as np
import subprocess, sys

DIR = "scenes"
df = pd.read_json("vlm3r.jsonl", lines=True)
scene_names = np.unique(df[df['data_source'] == 'scannet']['scene_name'])
for scene in scene_names:
    cmd = [
        sys.executable, "scannet.py",
        "-o", DIR,
        "--id", scene,
        "--type", ".sens",
    ]
    # Two newlines = accept both prompts
    subprocess.run(cmd, input="\n\n", text=True, check=True)