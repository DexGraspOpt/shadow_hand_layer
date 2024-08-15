# leap_hand layer for torch
import torch
import trimesh
import os
import numpy as np
import copy
import pytorch_kinematics as pk


import sys
sys.path.append('./')
from layer_asset_utils import save_part_mesh, sample_points_on_mesh, sample_visible_points

# All lengths are in mm and rotations in radians


class ShadowHandLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=True, show_mesh=False, hand_type='right', device='cuda'):
        super().__init__()
        self.BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
        self.show_mesh = show_mesh
        self.to_mano_frame = to_mano_frame
        self.device = device
        self.name = 'shadow_hand'
        self.hand_type = hand_type
        self.finger_num = 5

        urdf_path = os.path.join(self.BASE_DIR, '../assets/shadow_hand_{}.urdf'.format(hand_type))
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device)

        self.joints_lower = self.chain.low
        self.joints_upper = self.chain.high
        self.joints_mean = (self.joints_lower + self.joints_upper) / 2
        self.joints_range = self.joints_mean - self.joints_lower
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        # self.link_dict = {}
        # for link in self.chain.get_links():
        #     self.link_dict[link.name] = link.visuals[0].geom_param[0].split('/')[-1]
        #     self.scale = link.visuals[0].geom_param[1]

        # order in palm -> thumb -> index -> middle -> ring [-> pinky(little)]
        self.order_keys = [
            'palm',  # palm
            'thproximal', 'thmiddle', 'thdistal',  # thumb
            'ffproximal', 'ffmiddle', 'ffdistal',  # index
            'mfproximal', 'mfmiddle', 'mfdistal',  # middle
            'rfproximal', 'rfmiddle', 'rfdistal',  # ring
            'lfmetacarpal', 'lfproximal', 'lfmiddle', 'lfdistal',  # little
        ]

        self.ordered_finger_endeffort = ['palm',  'thdistal', 'ffdistal', 'mfdistal', 'rfdistal', 'lfdistal']

        # transformation for align the robot hand to mano hand frame, used for
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        if self.to_mano_frame:
            self.to_mano_transform[:3, :] = torch.tensor([[0.0, 0.0, -1.0, 0.099],
                                                         [0.0, 1.0, 0.0, 0.0],
                                                         [1.0, 0.0, 0.0, -0.011]])

        self.register_buffer('base_2_world', self.to_mano_transform)

        if not (os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_meshes_cvx')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_composite_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/visible_point_indices')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand.obj')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_all_zero.obj')
        ):
            # for first time run to generate contact points on the hand, set the self.make_contact_points=True
            self.make_contact_points = True
            self.create_assets()
        else:
            self.make_contact_points = False

        self.meshes = self.load_meshes()
        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
        theta = np.zeros((1, self.n_dofs), dtype=np.float32)

        save_part_mesh()
        sample_points_on_mesh()

        show_mesh = self.show_mesh
        self.show_mesh = True
        self.make_contact_points = True

        self.meshes = self.load_meshes()

        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        parts = mesh.split()

        new_mesh = trimesh.boolean.boolean_manifold(parts, 'union')
        new_mesh.export(os.path.join(self.BASE_DIR, '../assets/hand.obj'))

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(self.BASE_DIR, '../assets/hand_all_zero.obj'))

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes = self.load_meshes()

        self.get_forward_vertices(pose, theta)      # SAMPLE hand_composite_points
        sample_visible_points()

        self.show_mesh = True
        self.make_contact_points = False

        self.to_mano_transform[:3, :] = torch.tensor([[0.0, 0.0, -1.0, 0.099],
                                                      [0.0, 1.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, -0.011]])
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(self.BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        self.make_contact_points = False
        self.show_mesh = show_mesh

    def load_meshes(self):
        meshes = {}
        for link in self.chain.get_links():
            link_name = link.name
            if link_name not in self.order_keys:
                continue
            for visual in link.visuals:
                if visual.geom_type == None:
                    continue
                if visual.geom_type == 'mesh':
                    rel_path = visual.geom_param[0]
                    mesh_filepath = os.path.abspath(os.path.join(self.BASE_DIR, '../assets/', rel_path))
                    assert os.path.exists(mesh_filepath)
                    scale = visual.geom_param[1] if visual.geom_param[1] else [1.0, 1.0, 1.0]
                    link_pre_transform = visual.offset
                    mesh = trimesh.load(mesh_filepath, force='mesh')
                elif visual.geom_type == 'box':
                    mesh = trimesh.creation.box(extents=visual.geom_param)
                    scale = [1.0, 1.0, 1.0]
                    mesh_filepath = None
                else:
                    raise NotImplementedError

                if self.show_mesh:
                    if self.make_contact_points and mesh_filepath is not None:
                        mesh_filepath = mesh_filepath.replace('assets/hand_meshes/', 'assets/hand_meshes_cvx/').replace('.obj', '.stl')
                        mesh = trimesh.load(mesh_filepath, force='mesh')
                    mesh.apply_scale(scale)

                    verts = link_pre_transform.transform_points(torch.FloatTensor(np.array(mesh.vertices)))

                    temp = torch.ones(mesh.vertices.shape[0], 1).float()
                    vertex_normals = link_pre_transform.transform_normals(
                        torch.FloatTensor(copy.deepcopy(mesh.vertex_normals)))

                    meshes[link_name] = [
                        torch.cat((verts, temp), dim=-1).to(self.device),
                        mesh.faces,
                        torch.cat((vertex_normals, temp), dim=-1).to(self.device).to(torch.float)
                    ]
                else:
                    if mesh_filepath is not None:
                        vertex_path = mesh_filepath.replace('hand_meshes', 'hand_points').replace('.stl', '.npy').replace('.STL', '.npy').replace('.obj', '.npy')
                        assert os.path.exists(vertex_path)
                        points_info = np.load(vertex_path)
                    else:
                        print(visual, link_name)
                        raise NotImplementedError

                    if self.make_contact_points:
                        idxs = np.arange(len(points_info))
                    else:
                        idxs = np.load(os.path.dirname(os.path.realpath(__file__)) + '/../assets/visible_point_indices/{}.npy'.format(link_name))

                    verts = link_pre_transform.transform_points(torch.FloatTensor(points_info[idxs, :3]))
                    verts *= torch.tensor(scale, dtype=torch.float)

                    vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(points_info[idxs, 3:6]))

                    temp = torch.ones(idxs.shape[0], 1)

                    meshes[link_name] = [
                        torch.cat((verts, temp), dim=-1).to(self.device),
                        torch.zeros([0]),  # no real meaning, just for placeholder
                        torch.cat((vertex_normals, temp), dim=-1).to(torch.float).to(self.device)
                    ]

        return meshes

    def get_hand_segment_indices(self):
        hand_segment_indices = {}
        hand_finger_indices = {}
        segment_start = torch.tensor(0, dtype=torch.long, device=self.device)
        finger_start = torch.tensor(0, dtype=torch.long, device=self.device)
        for link_name in self.order_keys:
            end = torch.tensor(self.meshes[link_name][0].shape[0], dtype=torch.long, device=self.device) + segment_start
            hand_segment_indices[link_name] = [segment_start, end]
            if link_name in self.ordered_finger_endeffort:
                hand_finger_indices[link_name] = [finger_start, end]
                finger_start = end.clone()
            segment_start = end.clone()
        return hand_segment_indices, hand_finger_indices

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        ret = self.chain.forward_kinematics(theta)
        return ret

    def compute_abnormal_joint_loss(self, theta):
        loss_1 = torch.clamp(theta[:, 0] - theta[:, 4], 0, 1) * 10
        loss_2 = torch.clamp(theta[:, 4] + theta[:, 8], 0, 1) * 10
        loss_3 = -torch.clamp(theta[:, 8] - theta[:, 13], -1, 0) * 10
        loss_4 = torch.abs(theta[:, 12]) * 5
        loss_5 = torch.abs(theta[:, [2, 3, 6, 7, 10, 11,  15, 16]] - self.joints_mean[[2, 3, 6, 7, 10, 11, 15, 16]].unsqueeze(0)).sum(dim=-1) * 2
        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5

    def get_init_angle(self):
        init_angle = torch.tensor([-0.15, 0, 0.6, 0, 0, 0, 0.6, 0, -0.15, 0, 0.6, 0, 0, -0.25, 0, 0.6, 0,
                                        0, 1.2, 0, 0.0, 0], dtype=torch.float, device=self.device)
        return init_angle

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]
        meshes = []
        for key in self.order_keys:
            rotmat = ret[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            face = self.meshes[key][1]
            sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]

            meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand = [meshes[i][j] for i in range(len(meshes))]
            hand_mesh = np.sum(hand)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        outputs = self.forward(theta)

        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        outputs = self.forward(theta)

        verts = []
        verts_normal = []

        # for key, item in self.meshes.items():
        for key in self.order_keys:
            rotmat = outputs[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            vertex_normals = self.meshes[key][2]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts.append(batch_vertices)

            if self.make_contact_points:
                if not os.path.exists('../assets/hand_composite_points'):
                    os.makedirs('../assets/hand_composite_points', exist_ok=True)
                np.save('../assets/hand_composite_points/{}.npy'.format(key),
                        batch_vertices.squeeze().cpu().numpy())
            rotmat[:, :3, 3] *= 0
            batch_vertex_normals = torch.matmul(rotmat, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal


class ShadowAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            # thumb finger
            1369, 1551, 1569, 1655, 1584,
            922, 998,

            # index finger
            1707, 1861, 2019, 2020, 2005,
            167,

            # middle finger
            2068, 2222, 2380, 2381, 2366,
            562, 1132,

            # ring finger
            2429, 2583, 2741, 2742, 2727,
            2961, 960,  # 2961

            # little finger
            3050, 3204, 3362, 3363, 3348,

            # # plus side contact
            1865, 1838, 1848, 1837,
            2226, 2199, 2209, 2198,
            2587, 2560, 2570, 2559,
            3208, 3181,

        ])
        # vert_idx = np.load(os.path.join(BASE_DIR, 'anchor_idx.npy'))
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    hand = ShadowHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, hand.n_dofs), dtype=np.float32)
    # theta[0, 12:17] = np.array([-0.5, 0.5, 0, 0, 0])
    theta = torch.from_numpy(theta).to(device)
    # print(hand.joints_lower)
    # print(hand.joints_upper)
    theta = joint_angles_mu = torch.tensor([-0.15, 0, 0.6, 0, -0.15, 0, 0.6, 0, -0.15, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0,
                                        0, 1.2, 0, 0.0, 0], dtype=torch.float, device=device)

    # mesh version
    if show_mesh:
        mesh = hand.get_forward_hand_mesh(pose, theta)[0]
        mesh.show()
    else:
        # hand_segment_indices, hand_finger_indices = hand.get_hand_segment_indices()
        verts, normals = hand.get_forward_vertices(pose, theta)
        pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        mesh = trimesh.load(os.path.join(hand.BASE_DIR, '../assets/hand_to_mano_frame.obj'))
        # scene = trimesh.Scene([mesh, pc, ray_visualize])
        # scene.show()

        anchor_layer = ShadowAnchor()
        # anchor_layer.pick_points(verts.squeeze().cpu().numpy())
        anchors = anchor_layer(verts).squeeze().cpu().numpy()
        pc_anchors = trimesh.PointCloud(anchors, colors=(0, 0, 255))
        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[
                                                         0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        scene = trimesh.Scene([mesh, pc, pc_anchors, ray_visualize])
        scene.show()