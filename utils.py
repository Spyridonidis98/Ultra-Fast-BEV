from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, BitMap
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuimages import NuImages
import os 
import numpy as np
import cv2
from pyquaternion import Quaternion
import numpy as np
import math
import tensorflow as tf 


class NuScenesDataSet:
    def __init__(self, dataroot) -> None:
        self.dataroot = dataroot   
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.canvas_size = (1000, 1000) #height,width / y, x
        
        #vehicle corners in pixels
        ego_width = 2.1/2
        ego_length = 4.7/2
        ego_height = 1.6/2
        self.ego_bb = np.array([
            [ego_length, ego_width, -ego_height],  # Rear bottom left
            [ego_length, -ego_width, -ego_height],   # Rear bottom right
            [-ego_length, -ego_width, -ego_height],   # Front bottom left
            [-ego_length, ego_width, -ego_height],    # Front bottom right
            [ego_length, ego_width, ego_height],   # Rear top left
            [ego_length, -ego_width, ego_height],    # Rear top right
            [-ego_length, -ego_width, ego_height],    # Front top left
            [-ego_length, ego_width, ego_height]      # Front top right
        ]).transpose()
        self.ego_bb = np.concatenate((self.ego_bb, np.ones((1,8))), axis=0)

        #colors
        self.colors = [
            [1, 0, 0],  # Red
            [0, 0, 1],  # Blue
            [0, 0, 0],  # Black
            [0, 1, 0],  # Green
            [1, 0, 1],  # Magenta
            [1, 1, 0],  # Yellow
            [0.5, 0, 0],  # Brown
            [0.5, 0.5, 0],  # Olive
            [0, 0.5, 0],  # Dark Green
            [0.5, 0, 0.5],  # Purple
            [0, 0.5, 0.5],  # Teal
            [0, 0, 0.5],  # Navy
            [0.5, 0.5, 0.5],  # Gray
            [1, 1, 1]  # White
        ]

        #
        angle_rad = np.pi  
        self.rot_mat_180_z = np.eye(4)
        self.rot_mat_180_z[0:3, 0:3] = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    def get_transform_matrix(self, rotation, translation) -> np.ndarray:
        rotation_matrix = Quaternion(rotation).rotation_matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation
        return transformation_matrix



    def get_homogeneous_coordinates(sef, coords):
        coords = np.concatenate((coords, np.ones(shape=(1,coords.shape[1]))), axis= 0)
        return coords 

    def get_bb_from_vehicles(self, vehicles):
        #gets Bounding Box in local space of the vehicle 
        bb_list = [] 
        for vehicle in vehicles:
            dimensions = vehicle['dimensions']
            w, l, h = np.array(dimensions)/ 2.0
            corners = np.array([
                [l, w, -h],
                [l, -w, -h],
                [-l, -w, -h],
                [-l, w, -h],
                [l, w, h],
                [l, -w, h],
                [-l, -w, h],
                [-l, w, h]
            ]).transpose()
            bb_list.append(corners)
        return bb_list
    
    def get_vehicles_from_sample(self, sample):
        vehicles = []
        # Iterate over the sample's annotations.
        for annotation_token in sample['anns']:
            # Get the annotation data.
            annotation = self.nusc.get('sample_annotation', annotation_token)

            # Check if the annotation is a vehicle.
            if 'vehicle' in annotation['category_name']:
                # if annotation['visibility_token'] in ('1'):continue #visibility of object is under 40% skip
                # Get the vehicle's position, dimensions, and rotation.
                vehicle_position = annotation['translation']
                vehicle_dimensions = annotation['size']
                vehicle_rotation = annotation['rotation']

                # Add the vehicle's data to our list.
                vehicles.append({
                    'translation': vehicle_position,
                    'dimensions': vehicle_dimensions,
                    'rotation': vehicle_rotation
                })
        return vehicles
    
    def from_mask_to_rgb(self, mask):
        rgb_map_mask = np.zeros(shape = (mask.shape[0], mask.shape[1], 3), dtype=np.float64)
        for c in range(mask.shape[2]):
            for i in range(3):
                rgb_map_mask[:, :, i] += mask[:, :, c] * self.colors[c][i]
        return rgb_map_mask
    
    def bb_coords_to_mask(self, vehicles_bb, distance_around_ego = (100,100)):
        vehicles_bb = np.copy(vehicles_bb)
        vehicles_bb_uv = []
        mask = np.zeros(shape = (self.canvas_size[0], self.canvas_size[1], 3))
        
        for bb in vehicles_bb:
            bb = bb[0:2,0:4]
            bb[0, :] = bb[0, :] * self.canvas_size[0] / distance_around_ego[0] + self.canvas_size[0]/2
            bb[1, :] = bb[1, :] * self.canvas_size[1] / distance_around_ego[1] + self.canvas_size[1]/2
            bb = np.flip(bb, axis=0)
            bb = np.transpose(bb.astype(np.int32))
            vehicles_bb_uv.append(bb)
        
        #the input for fillpoly is a list of shapes and each shape has an np.array of shape(points, point_axis(x,y))
        #the x axis for the fill poly function is looking right and the y axis is looking down 
        mask = cv2.fillPoly(mask, pts=vehicles_bb_uv, color=(1,1,1))
        return mask[:,:,0:1].astype(np.uint8)

    def lidar_points_to_mask(self, lps, distance_around_ego = (100,100)):
        lps = np.copy(lps)
        mask = np.zeros(shape = (self.canvas_size[0], self.canvas_size[1], 1), dtype=np.uint8)
        lps[0, :] = lps[0, :] * self.canvas_size[0] / distance_around_ego[0] + self.canvas_size[0]/2
        lps[1, :] = lps[1, :] * self.canvas_size[1] / distance_around_ego[1] + self.canvas_size[1]/2 #the y that we get from get_map_mask() is upside down realtive to the map so we have to
        
        for i in range(lps.shape[1]):
                if lps[0, i] > 0 and lps[0, i] < self.canvas_size[0] and lps[1, i] > 0 and lps[1, i] < self.canvas_size[1]:
                    mask[int(lps[0, i]), int(lps[1, i])] = 1

        return mask
    
    def clean_mask(self, mask):
        #some masks overlap with each other, we clean them so there is no overlap
        for i in range(mask.shape[-1]):
            for i2 in range(i+1, mask.shape[-1]):
                mask[:, :, i] = (mask[:, :, i] - mask[:, :, i2])>0
            
        return mask

    def get_samples_bev_mask(self, sample, distance_around_ego = (100,100), layers_names = ['drivable_area'], mask_size = (200,200)):
        #get scene of the sample and map
        scene = self.nusc.get('scene', sample['scene_token'])
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=self.nusc.get('log', scene['log_token'])['location'])

        sample_data_list = {}
        for data_key in sample['data']:
            if data_key[0:5] == 'RADAR':continue
            sample_data_list[data_key] = self.nusc.get('sample_data', sample['data'][data_key])

        ego_pose = self.nusc.get('ego_pose', sample_data_list['LIDAR_TOP']['ego_pose_token'])
        patch_box = [ego_pose['translation'][0], ego_pose['translation'][1], distance_around_ego[0], distance_around_ego[1]]
        patch_angle = math.degrees(Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]) - 90
        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layers_names, self.canvas_size[0:2])
        map_mask = map_mask.transpose(1,2,0)
        map_mask = np.flip(map_mask, axis=0)#flips the x axis 

        vehicles = self.get_vehicles_from_sample(sample)
        vehicles_bb_local = self.get_bb_from_vehicles(vehicles)
        velicle2world = [self.get_transform_matrix(vehicle['rotation'], vehicle['translation']) for vehicle in vehicles]
        vehicle_bb_world = [np.dot(velicle2world[i],self.get_homogeneous_coordinates(vehicles_bb_local[i])) for i in range(len(vehicles))]
        ego2world = self.get_transform_matrix(ego_pose['rotation'], ego_pose['translation'])
        world2ego = np.linalg.inv(ego2world)
        vehicle_bb_ego = [np.dot(world2ego, vehicle_bb_world[i]) for i in range(len(vehicles))]
        # vehicle_bb_ego.append(self.ego_bb)
        vehicle_bb_ego = [np.dot(self.rot_mat_180_z, vehicle_bb_ego[i]) for i in range(len(vehicles))]#rotate 180, len(vehicles)+1 if we  want to add ego vehicle
        vehicles_mask = self.bb_coords_to_mask(vehicle_bb_ego, distance_around_ego)
        map_mask = np.concatenate((map_mask, vehicles_mask), axis=2)

        map_mask = map_mask.astype(np.int32)
        map_mask = self.clean_mask(map_mask)
        map_mask_nothings =  (np.ones(map_mask.shape[0:2], dtype=np.int32) - map_mask.sum(axis = 2))>0
        map_mask_nothings = np.expand_dims(map_mask_nothings, axis=2)
        map_mask = np.concatenate((map_mask, map_mask_nothings), axis=2)
        map_mask = map_mask.astype(np.float32)

        map_mask = cv2.resize(map_mask, mask_size, interpolation=cv2.INTER_AREA)
        map_mask = tf.one_hot(np.argmax(map_mask, axis=-1).astype(np.int32), map_mask.shape[-1], dtype=tf.float32).numpy()

        return map_mask

    def add_ego_vehicle_to_mask(self, mask, distance_around_ego = (100,100)):
        ego_vehicle_mask = np.zeros(shape = (*mask.shape[:-1],1), dtype=np.float32)
        vehicle_bb_ego = [self.ego_bb]
        ego_vehicle_mask = self.bb_coords_to_mask(vehicle_bb_ego, distance_around_ego)
        ego_vehicle_mask = cv2.resize(ego_vehicle_mask, mask.shape[:-1], interpolation=cv2.INTER_AREA)[..., np.newaxis]
        ego_vehicle_mask = (ego_vehicle_mask>0.5)
        mask = np.concatenate((mask, ego_vehicle_mask),axis=-1)
        return mask


    def get_samples_camera_data(self, sample):
        sample_data_list = {}
        for data_key in sample['data']:
            if 'CAM' not in data_key:continue
            sample_data_list[data_key] = self.nusc.get('sample_data', sample['data'][data_key])
        
        images = {}
        for data_key in sample_data_list:
            file_path = sample_data_list[data_key]['filename']
            file_path = os.path.join(self.dataroot, file_path)
            images[data_key] = np.flip(cv2.imread(file_path), axis=2)
        
        return images
    

    def get_samples_lidar_points_projected_to_camera(self, sample):
        #returns the lidar points from the perspective of the camera in a form of (axis, point) shape = (2, points)
        #the x axis of the points is looking right and the y axis down, the same as the camera axis 
        #you can use plt.scatter(points[0,:] -> x, points[1,:] -> y) to plot the points
        #plt scatter has its x axis looking right and y axis looking down, the same as our points 
        
        cameras_projection_matrix = {}#from coordinates relative to the camera to uv coordinates projected to the camera
        camera2ego = {} #from cemeras point of reference to ego vehicles point of reference transformation matrix 
        for data_key in sample['data']:
            if 'CAM' not in data_key:continue
            sample_data = self.nusc.get('sample_data', sample['data'][data_key])
            camera_calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            camera2ego[data_key] = self.get_transform_matrix(camera_calibrated_sensor['rotation'], camera_calibrated_sensor['translation'])
            cameras_projection_matrix[data_key] = np.array(camera_calibrated_sensor['camera_intrinsic'], dtype=np.float32)

        lidar_sample_data =  self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        lidar2ego =  self.get_transform_matrix(lidar_calibrated_sensor['rotation'], lidar_calibrated_sensor['translation'])
        lidar2cam = {}
        for data_key in cameras_projection_matrix:
            ego2camera = np.linalg.inv(camera2ego[data_key])
            lidar2cam[data_key] = np.dot(ego2camera, lidar2ego)    

        lidar_points_camera_uv = {}#lidar points maped to the cameras uv coordinates
        lidar_points = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_sample_data['token'])).points
        for data_key in lidar2cam:
            lidar_points_camera = np.dot(lidar2cam[data_key], self.get_homogeneous_coordinates(lidar_points[0:3, :]))
            lidar_points_uv = np.dot(cameras_projection_matrix[data_key], lidar_points_camera[0:3, :])
            lidar_points_uv[0:2, :] = lidar_points_uv[0:2, :]/lidar_points_uv[2:3, :]#divide by z
            lp_filter = np.ones(lidar_points_uv.shape[1], dtype=bool)
            lp_filter = np.logical_and(lp_filter, lidar_points_uv[2, :] > 1)#discard points closer than 1m
            lp_filter = np.logical_and(lp_filter, lidar_points_uv[0, :] > 0)
            lp_filter = np.logical_and(lp_filter, lidar_points_uv[0, :] < 1600 - 1)
            lp_filter = np.logical_and(lp_filter, lidar_points_uv[1, :] > 0)
            lp_filter = np.logical_and(lp_filter, lidar_points_uv[1, :] < 900 - 1)
            lidar_points_uv = lidar_points_uv[:, lp_filter]
            lidar_points_intesity = lidar_points[3:4, lp_filter]
            lidar_points_camera_uv[data_key] = np.concatenate((lidar_points_uv[0:3, :], lidar_points_intesity), axis=0)
        

        return lidar_points_camera_uv #(u,v,z,intensity) 
    
    def get_samples_features_projected_from_camera_to_ego(self, sample, distance=10, step=1):
        
        cameras_projection_matrix_inv = {}#from coordinates relative to the camera to uv coordinates projected to the camera
        camera2ego = {} #from cemeras point of reference to ego vehicles point of reference transformation matrix 
        for data_key in sample['data']:
            if 'CAM' not in data_key:continue
            sample_data = self.nusc.get('sample_data', sample['data'][data_key])
            camera_calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            camera2ego[data_key] = self.get_transform_matrix(camera_calibrated_sensor['rotation'], camera_calibrated_sensor['translation'])
            cameras_projection_matrix_inv[data_key] = np.linalg.inv(np.array(camera_calibrated_sensor['camera_intrinsic'], dtype=np.float32))

        zdot = 0
        dot = []
        features_projected = {}
        while zdot < distance:
            zdot+=step
            for u in range(1600-1):
                # for v in range(900-1):
                    if u % 16 != 0: continue 
                    v = 450
                    xdot = u*zdot
                    ydot = v*zdot
                    dot.append([xdot, ydot, zdot])
        dot = np.array(dot, dtype=np.float32).transpose()


        for data_key in cameras_projection_matrix_inv:
            proj_inv = cameras_projection_matrix_inv[data_key]
            features_projected_camera = np.dot(proj_inv, dot)
            features_projected_ego = np.dot(camera2ego[data_key], self.get_homogeneous_coordinates(features_projected_camera))
            features_projected_ego = np.dot(self.rot_mat_180_z, features_projected_ego)
            features_projected[data_key] = features_projected_ego
        

        return features_projected


    def create_images_LUTs(self, voxel_size=(200,200,4), image_size=(56,100), metric_span = [(-50,50),(-50,50),(-1,3)]):
        #get projection matrix for each camera and ego2camera transformation matrices
        sample = self.nusc.sample[0]
        cameras_projection_matrix = {}#from coordinates relative to the camera to uv coordinates projected to the camera
        camera2ego = {} #from cemeras point of reference to ego vehicles point of reference transformation matrix 
        for data_key in sample['data']:
            if 'CAM' not in data_key:continue
            sample_data = self.nusc.get('sample_data', sample['data'][data_key])
            camera_calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            camera2ego[data_key] = self.get_transform_matrix(camera_calibrated_sensor['rotation'], camera_calibrated_sensor['translation'])
            cameras_projection_matrix[data_key] = np.array(camera_calibrated_sensor['camera_intrinsic'], dtype=np.float32)
        ego2camera = {}
        for data_key in camera2ego:
            ego2camera[data_key] = np.linalg.inv(camera2ego[data_key])

        #create voxel with coordinates relative to ego 
        distance_around_ego = [span[1]-span[0] for span in metric_span]
        voxel_ego = np.zeros(voxel_size+(3,), dtype=np.float32)
        voxel_indices = np.zeros(voxel_size+(3,), dtype=np.int32)
        for x in range(voxel_size[0]):
            for y in range(voxel_size[1]):
                for z in range(voxel_size[2]):
                    voxel_indices[x,y,z,:] = [x,y,z]

                    xe = - (x * distance_around_ego[0])/(voxel_size[0]-1) + metric_span[0][1]
                    ye = - (y * distance_around_ego[1])/(voxel_size[1]-1) + metric_span[1][1]
                    ze = - (z * distance_around_ego[2])/(voxel_size[2]-1) + metric_span[2][1]

                    voxel_ego[x,y,z,:] = [xe, ye ,ze]
        voxel_ego_flat = voxel_ego.reshape(-1,3)
        voxel_ego_flatT = self.get_homogeneous_coordinates(voxel_ego_flat.T)
        voxel_indices_flatT = voxel_indices.reshape(-1,3).T

                                 
        #get LUTs for each camera
        original_image_size = (900, 1600)
        cameras_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"] #must be the same order as in the data pipeline
        images_LUTs = []
        for data_key in cameras_order:
            voxel_camera = np.matmul(ego2camera[data_key], voxel_ego_flatT)#cordinates relative to camera
            uv = np.matmul(cameras_projection_matrix[data_key], voxel_camera[0:3])
            uv[0:2] = uv[0:2]/uv[2]
            uv[0] *= image_size[1]/original_image_size[1]
            uv[1] *= image_size[0]/original_image_size[0]
            lp_filter = np.ones(uv.shape[1], dtype=bool)
            lp_filter = np.logical_and(lp_filter, uv[2, :] > 0.5)#discard points closer than 0.5m
            lp_filter = np.logical_and(lp_filter, uv[0, :] > 0)
            lp_filter = np.logical_and(lp_filter, uv[0, :] < image_size[1] - 1)#u -> width -> y
            lp_filter = np.logical_and(lp_filter, uv[1, :] > 0)
            lp_filter = np.logical_and(lp_filter, uv[1, :] < image_size[0] - 1)#v -> height -> x
            uv = uv[:, lp_filter]
            vu = np.flip(uv[0:2], axis=0).astype(np.int32)# u -> y, v -> x. so we have to flip the axis to use as indices
            vi = voxel_indices_flatT[:, lp_filter].astype(np.int32)
            images_LUTs.append([tf.constant(vu.T), tf.constant(vi.T)])

        return images_LUTs #different cameras LUTs

    def get_samples_bev_lidar(self, sample, voxel_size = (200,200,4), metric_span = [(-50,50),(-50,50),(-1,3)]):
        
        lidar_sample_data =  self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        lidar2ego =  self.get_transform_matrix(lidar_calibrated_sensor['rotation'], lidar_calibrated_sensor['translation'])
        lidar_points = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_sample_data['token'])).points
        lidar_points = self.get_homogeneous_coordinates(lidar_points[0:3, :])
        lidar_points_ego = np.dot(lidar2ego, lidar_points)

        distance_around_ego = [span[1]-span[0] for span in metric_span]

        lidar_points_voxel = np.zeros(shape=lidar_points_ego.shape, dtype=np.float32)
        lidar_points_voxel[0] = -lidar_points_ego[0] * voxel_size[0] / distance_around_ego[0] + voxel_size[0] * metric_span[0][1]/distance_around_ego[0]
        lidar_points_voxel[1] = -lidar_points_ego[1] * voxel_size[1] / distance_around_ego[1] + voxel_size[1] * metric_span[1][1]/distance_around_ego[1]
        lidar_points_voxel[2] = -lidar_points_ego[2] * voxel_size[2] / distance_around_ego[2] + voxel_size[2] * metric_span[2][1]/distance_around_ego[2]

        lp_filter = np.ones(lidar_points_voxel.shape[1], dtype=bool)
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[0, :] >=0)#x axis 
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[0, :] < voxel_size[0])
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[1, :] >=0)#y axis 
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[1, :] < voxel_size[1])
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[2, :] >=0)#z axis 
        lp_filter = np.logical_and(lp_filter, lidar_points_voxel[2, :] < voxel_size[2])
        lidar_points_voxel = lidar_points_voxel[:, lp_filter][0:3, :]

        voxel = np.zeros(voxel_size, dtype=np.float32)
        for i in range(lidar_points_voxel.shape[1]):
            px = int(lidar_points_voxel[0, i])
            py = int(lidar_points_voxel[1, i])
            pz = int(lidar_points_voxel[2, i])
            voxel[px,py,pz] = 1
    
        return voxel

        #lidar points relative to voxel

    @staticmethod
    def save_as_binary(array, file_path):
        """
        :param array: must be an numpy array.
        """
        array_as_bytes = array.tobytes()
        tf.io.write_file(
            filename= file_path,
            contents= array_as_bytes,
        )

    @staticmethod
    def read_binary_file(file_path, dtype, shape):
        binary_data = tf.io.read_file(file_path)
        array = tf.io.decode_raw(binary_data, dtype)
        array = tf.reshape(array, shape=shape)
        return array


    def get_training_val_samples(self):
        scenes_split_names = create_splits_scenes()
        scenes = self.nusc.scene
        train_samples = []
        val_samples = []

        for scene_name in scenes_split_names['train']:
            for scene in scenes:
                if scene_name == scene['name']:
                    sample = self.nusc.get('sample', scene['first_sample_token'])
                    train_samples.append(sample)
                    while True:
                        if sample['next']=='':
                            break
                        sample = self.nusc.get('sample', sample['next'])
                        train_samples.append(sample)
        
        for scene_name in scenes_split_names['val']:
            for scene in scenes:
                if scene_name == scene['name']:
                    sample = self.nusc.get('sample', scene['first_sample_token'])
                    val_samples.append(sample)
                    while True:
                        if sample['next']=='':
                            break
                        sample = self.nusc.get('sample', sample['next'])
                        val_samples.append(sample)

        return train_samples, val_samples


    def get_train_val_cameras_file_paths(self):
        """
        sensors paths order 
        [
            ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
            ,...
        ]
        """
        train_samples, val_samples = self.get_training_val_samples()
        train_file_paths = []
        val_file_paths = []

        for sample in train_samples:
            camera_path = {}
            for data_key in sample['data']:
                if 'CAM' not in data_key:continue
                file_path = self.nusc.get('sample_data', sample['data'][data_key])['filename']
                file_path = os.path.join(self.dataroot, file_path)
                camera_path[data_key] = file_path
            train_file_paths.append([camera_path['CAM_FRONT'], camera_path['CAM_FRONT_LEFT'], camera_path['CAM_FRONT_RIGHT'], camera_path['CAM_BACK'], camera_path['CAM_BACK_LEFT'], camera_path['CAM_BACK_RIGHT']])
        
        for sample in val_samples:
            camera_path = {}
            for data_key in sample['data']:
                if 'CAM' not in data_key:continue
                file_path = self.nusc.get('sample_data', sample['data'][data_key])['filename']
                file_path = os.path.join(self.dataroot, file_path)
                camera_path[data_key] = file_path
            val_file_paths.append([camera_path['CAM_FRONT'], camera_path['CAM_FRONT_LEFT'], camera_path['CAM_FRONT_RIGHT'], camera_path['CAM_BACK'], camera_path['CAM_BACK_LEFT'], camera_path['CAM_BACK_RIGHT']])

        return np.array(train_file_paths), np.array(val_file_paths)
    
    def get_day_night_val_samples_indexes(self):
        val_samples = self.get_training_val_samples()[1]
        index_val_samples_day = []
        index_val_samples_night = []
        for index, sample in enumerate(val_samples):
            scene_description = self.nusc.get('scene', sample['scene_token'])['description']
            if 'Night' in scene_description:
                index_val_samples_night.append(index)
            else:
                index_val_samples_day.append(index)
        return np.array(index_val_samples_day, dtype=np.int32), np.array(index_val_samples_night, np.int32)

    @staticmethod
    def get_train_val_file_paths(file_path_train, file_path_val):

        train_files_paths = [os.path.join(file_path_train, file) for file in os.listdir(file_path_train)]
        val_files_paths = [os.path.join(file_path_val, file) for file in os.listdir(file_path_val)]

        train_files_paths = sorted(train_files_paths, key = lambda file_path: int(file_path.split('/')[-1].split('.')[0]) )
        val_files_paths = sorted(val_files_paths, key = lambda file_path: int(file_path.split('/')[-1].split('.')[0]) )

        return train_files_paths, val_files_paths 



class NuImagesDataSet:
    def __init__(self, dataroot) -> None:
        self.nuim_train = NuImages(dataroot=dataroot, version='v1.0-train', verbose=True, lazy=True)
        self.nuim_val = NuImages(dataroot=dataroot, version='v1.0-val', verbose=True, lazy=True)
        self.dataroot = dataroot 

    def get_cameras_train_val_file_paths(self):
        train_file_paths = []
        val_file_paths = []
        for sample in self.nuim_train.sample:
            train_file_paths.append(self.dataroot+'/'+self.nuim_train.get('sample_data', sample['key_camera_token'])['filename'])
        for sample in self.nuim_val.sample:
            val_file_paths.append(self.dataroot+'/'+self.nuim_val.get('sample_data', sample['key_camera_token'])['filename'])
        
        return train_file_paths, val_file_paths
    

    def get_samples_mask(self, sample, size=(56,100), version=None):
        assert version=='train' or version=='val', 'version must be train or val'
        if version == 'train':nuim = self.nuim_train
        else: nuim = self.nuim_val

        key_camera_token = sample['key_camera_token']
        dataroot =  self.dataroot + '/' +nuim.get('sample_data', key_camera_token)['filename']
        semantic_mask, instance_mask = nuim.get_segmentation(key_camera_token)
        mask = np.zeros((semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.int32)
        mask[:, :, 0] = (semantic_mask==24) #road
        mask[:, :, 1] = np.logical_and(semantic_mask>=14, semantic_mask<=23) #vehicles
        mask[:, :, 2] = (mask[:, :, 2] == (mask[:, :, 0]+mask[:, :, 1])) #nothing-no class, what dosent belong to road or vehicles
        mask = mask.astype(np.float32)
        if size != (900,1600):
            mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_AREA) #INTER_NEAREST -> We get artifacts
            mask = tf.one_hot(np.argmax(mask, axis=-1).astype(np.int32), mask.shape[-1], dtype=tf.float32).numpy()
        return mask.astype(np.float32)

    @staticmethod
    def save_as_binary(array, file_path):
        """
        :param array: must be an numpy array.
        """
        array_as_bytes = array.tobytes()
        tf.io.write_file(
            filename= file_path,
            contents= array_as_bytes,
        )

    @staticmethod
    def read_binary_file(file_path, dtype, shape):
        binary_data = tf.io.read_file(file_path)
        array = tf.io.decode_raw(binary_data, dtype)
        array = tf.reshape(array, shape=shape)
        return array
    
    @staticmethod
    def get_train_val_file_paths(file_path_train, file_path_val):

        train_files_paths = [os.path.join(file_path_train, file) for file in os.listdir(file_path_train)]
        val_files_paths = [os.path.join(file_path_val, file) for file in os.listdir(file_path_val)]

        train_files_paths = sorted(train_files_paths, key = lambda file_path: int(file_path.split('/')[-1].split('.')[0]) )
        val_files_paths = sorted(val_files_paths, key = lambda file_path: int(file_path.split('/')[-1].split('.')[0]) )

        return train_files_paths, val_files_paths 

    @staticmethod
    def get_image(image_path):
        # scale = 1/255.0
        image_shape = (448, 800)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_shape)#*scale
        #image= tf.cast(image, tf.float16)
        return image
    
'''
Extra:
NuimagesDataSet

get_samples_mask:
    segmentation mask: datatype=numpy.array, shape = (x,y), dtype=float32.
    Values of segmentation mask are integer number where each different number represent a different class.
    Use name_to_index_mapping(nuimd.nuim.category)
    to get what categories are mapped to what ids in the segmentation mask.

    {'no class':0 #added by me, it is not retirned by name_to_index_mapping() 
    'animal': 1,
    'human.pedestrian.adult': 2,
    'human.pedestrian.child': 3,
    'human.pedestrian.construction_worker': 4,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 6,
    'human.pedestrian.stroller': 7,
    'human.pedestrian.wheelchair': 8,
    'movable_object.barrier': 9,
    'movable_object.debris': 10,
    'movable_object.pushable_pullable': 11,
    'movable_object.trafficcone': 12,
    'static_object.bicycle_rack': 13,
    'vehicle.bicycle': 14,
    'vehicle.bus.bendy': 15,
    'vehicle.bus.rigid': 16,
    'vehicle.car': 17,
    'vehicle.construction': 18,
    'vehicle.emergency.ambulance': 19,
    'vehicle.emergency.police': 20,
    'vehicle.motorcycle': 21,
    'vehicle.trailer': 22,
    'vehicle.truck': 23,
    'flat.driveable_surface': 24,
    'vehicle.ego': 31}
    

'''