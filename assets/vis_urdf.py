from urdfpy import URDF

robot = URDF.load('./shadow_hand_right.urdf')
cfg = {}

fk = robot.visual_trimesh_fk(cfg)
# robot.show(cfg=cfg, use_collision=False)


# import os
#
# import trimesh
#
# for root, dirs, files in os.walk('./hand_meshes'):
#     for filename in files:
#         filepath = os.path.join(root, filename)
#         mesh = trimesh.load(filepath)
#         mesh.show()
#         new_filepath = filepath.replace('.STL', '.stl')
#         mesh.export(new_filepath)
