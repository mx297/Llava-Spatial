import os

with open("/home/mohamed.abouelhadid/Llava-Spatial/preprocessing/preprocess_scannet/splits/scannetv2_train.txt","r") as f:
    train_scenes = set(map(str.strip,f.readlines()))

color_sampled_scenes = set(os.listdir("/l/users/mohamed.abouelhadid/sampled/intrinsic/train"))
print(len(color_sampled_scenes))
video_sampled_scenes = set(os.listdir("/l/users/mohamed.abouelhadid/videos/output/train"))
video_sampled_scenes = [v[:-4] for v in video_sampled_scenes]

mesh_sampled_scenes = set(os.listdir("/l/users/mohamed.abouelhadid/scannet200/output/train"))
mesh_sampled_scenes = [m[:-4] for m in mesh_sampled_scenes]

color_diff = train_scenes.difference(color_sampled_scenes)
video_diff = train_scenes.difference(video_sampled_scenes)
mesh_diff = train_scenes.difference(mesh_sampled_scenes)

#print(color_diff)
print(video_diff)
#print(mesh_diff)
