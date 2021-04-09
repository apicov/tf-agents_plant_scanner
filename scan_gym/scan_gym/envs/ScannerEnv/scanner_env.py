import numpy as np
#import cv2
from os import listdir
from os.path import isfile, join
import gym
import glob
from PIL import Image
import open3d as o3d
from .cl import *
from skimage.morphology import binary_dilation
from .proc3d import *
import json
from .utils import *
import glob
import os

KEEP_IMAGE = 22
CLOCKWISE = 20
ANTICLOCKWISE = 21


class ScannerEnv(gym.Env):
    """
    A template to implement custom OpenAI Gym environments
    """
    metadata = {'render.modes': ['human']}
    def __init__(self,dataset_path,init_pos_inc_rst=False):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        self.n_positions = 720 #total of posible positions in env
        self.n_zones = 8 #number of zones by which the circle is divided
        self.max_temp_moves_count = 20
        self.previous_ref_img = 0
        self.previous_distance_to_ref_img=0
        self.total_moves = 0
        self.init_pos_inc_rst = init_pos_inc_rst #if false init position is random, if true, it starts in position 0 and increments by 1 position every reset
        self.init_pos_counter = 0


        self.obs_images = self.load_images(os.path.join(dataset_path, 'gray_imgs_82'),'png')
        self.rwd_images = self.load_images(os.path.join(dataset_path, 'imgs'),'png')
        self.extrinsics = self.load_extrinsics(os.path.join(dataset_path, 'extrinsics'))
        self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.camera_model = json.load(open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics= self.camera_model['params'][0:4]
        self.chamfer_limits = json.load(open(os.path.join(dataset_path, 'chamfer_distance_gt_limits.json'))) #minimum and maximun chamfer distances calculated with ground truth model
        
        params = json.load(open(os.path.join(dataset_path, 'params.json')))
        self.gt=o3d.io.read_point_cloud(params["gt_path"])
        self.gt_points = np.asarray(self.gt.points)
        self.n_dilation=params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']
        
        
        self.set_sc(self.bbox)

        self.im_ob_space = gym.spaces.Box(low=0, high=255, shape=(82,82,2))#, dtype=np.uint8)

        #[distance from closest saved image, covered area (self.n_zones sections), number of actions executed since last save image action,
        # delta distance (diff between las and current distance from closest saved image), 
        #direction of movement (clockwise or anticlockwise)]                                           
        lowl = np.array([-400,0,0,-20,0])
        highl = np.array([400,self.n_zones,self.max_temp_moves_count,20,1])                                           
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.float32)
        self.observation_space = gym.spaces.Tuple((self.im_ob_space, self.vec_ob_space))
        
        # Modify the action space, and dimension according to your custom
        #environment's needs
        self.action_space = gym.spaces.Discrete(23)
        self.done = False
        self.current_state = ((),())
        self.current_position = 0
        self.previous_state = ((),())
        self.previous_action = None
        #self._spec.id = "Romi-v0"
        self.num_steps = 0
        self.total_reward = 0
        
        self.kept_images = []
        self.dir = 0
        
        #every zone is  images long ( degress in this case)
        #1 means that there is already a kept pixture in that zone
        self.covered_zones = np.zeros(self.n_zones)
        self.covered_area = 0

        #writes a 1 on the index of corresponding kept image
        self.kept_im_map = np.zeros(self.n_positions,dtype='uint')
        

    @property
    def nA(self):
        return self.action_space.n

    def reset(self):
        self.covered_zones = np.zeros(self.n_zones)
        self.num_steps = 0
        self.total_reward = 0
        self.done = False
        self.previous_state = ((),())
        self.total_moves = 0
        
        if self.init_pos_inc_rst :
            self.current_position = self.init_pos_counter
            self.init_pos_counter += 1
            if self.init_pos_counter >= self.n_positions:
                self.init_pos_counter = 0      
        else:
            self.current_position =  np.random.randint(0,self.n_positions)

        self.dir =  np.random.randint(2)

        #the image at the beginning position is always kept
        self.previous_action = KEEP_IMAGE
        self.kept_images = []
        self.kept_images.append(self.current_position)

        area_idx = self.get_area_section_n(self.current_position)
        self.covered_zones[area_idx] = 1
        self.covered_area = 1

        self.temp_moves_count = 0

        self.previous_ref_img = self.current_position
        self.previous_distance_to_ref_img=0

        self.kept_im_map = np.zeros(self.n_positions,dtype='uint')
        self.kept_im_map[self.current_position] = 1

        #print(self.previous_match_idx,self.match_idx)
        self.current_state = ( np.dstack( (self.obs_images[self.current_position],self.obs_images[self.current_position])) , (0, self.covered_area,self.temp_moves_count,0,self.dir))
        return self.current_state
     
        
    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return

    def step(self, action):
        self.exec_action(action)
        reward = self.calculate_reward(action)
        self.previous_action = action
        
        if self.temp_moves_count >= self.max_temp_moves_count:
            if self.covered_area>=self.n_zones:
                cd=self.chamfer_from_collected()
                reward+= self.minMaxNorm(cd,self.chamfer_limits['max'],self.chamfer_limits['min'],0,500)
                self.done = True
            else:
                reward += -5
            	
        if self.num_steps >= 2000:
            self.done = True
            if self.covered_area>=self.n_zones:
                cd=self.chamfer_from_collected()
                reward+= self.minMaxNorm(cd,self.chamfer_limits['max'],self.chamfer_limits['min'],0,500)
                print(cd)
            else:
                reward -= 100
           
           
        self.total_reward += reward
        self.num_steps += 1

        return self.current_state, reward, self.done, {}


    def chamfer_from_collected(self):
        del(self.sc)
        self.set_sc(self.bbox) 
        for i in self.kept_images:
            self.carve(i)
        ch_dist = self.dist_to_gt()
        return ch_dist

    def exec_action(self,action):       
        #keep picture and sum distance
        if action == KEEP_IMAGE:
            self.kept_images.append(self.current_position)
            
            area_idx = self.get_area_section_n(self.current_position)
            if self.covered_zones[area_idx] == 0:
                self.covered_area += 1
                self.covered_zones[area_idx] = 1

            self.temp_moves_count = 0

            self.previous_ref_img = self.current_position
            self.previous_distance_to_ref_img=0

            self.kept_im_map[self.current_position] = 1
           
            self.previous_state = self.current_state

            self.current_state = ( np.dstack( (self.obs_images[self.current_position],self.obs_images[self.current_position])) ,(0,self.covered_area,self.temp_moves_count,0,self.dir))

        elif action == CLOCKWISE:
            self.dir = 0
            self.temp_moves_count = min( self.temp_moves_count + 1,  self.max_temp_moves_count)
            self.current_state = (self.current_state[0], (self.current_state[1][0],self.current_state[1][1],self.temp_moves_count,self.current_state[1][3],self.dir))

        elif action == ANTICLOCKWISE:
            self.dir = 1
            self.temp_moves_count = min( self.temp_moves_count + 1,  self.max_temp_moves_count)
            self.current_state = (self.current_state[0], (self.current_state[1][0],self.current_state[1][1],self.temp_moves_count,self.current_state[1][3],self.dir))
                
        else:
            #move #set new position
            steps = self.action2move(action,self.dir)
            self.current_position = self.calculate_position(self.current_position,steps)

            closest_kept_img, distance_to_closest_img = self.get_closest_img(self.current_position)
            
            area_idx = self.get_area_section_n(self.current_position)
            
            self.temp_moves_count = min( self.temp_moves_count + 1,  self.max_temp_moves_count)

            if closest_kept_img == self.previous_ref_img:#if the closest kept image continues to be the same
                delta_distance_to_ref = distance_to_closest_img - self.previous_distance_to_ref_img
            else: #there is a new closest kept image
                delta_distance_to_ref = 0
                self.previous_ref_img = closest_kept_img

            self.previous_distance_to_ref_img = distance_to_closest_img
            self.total_moves += 1
            
            self.previous_state = self.current_state
            self.current_state = (  np.dstack( (self.obs_images[closest_kept_img],self.obs_images[self.current_position]))  ,(distance_to_closest_img,self.covered_area,self.temp_moves_count,delta_distance_to_ref,self.dir) )
            


    def calculate_reward(self,action):
        #if camera was moved
        if action < 20:
            reward = -1.0
        elif action == KEEP_IMAGE:
            reward = -0.5
        else:
            reward = 0
        return reward


 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin

    '''    
    #maps numerical movement to correspondend action number
    def move2action(self,move):
        if move>=1 and move<21:
            return move-1
        elif move>=-20 and move<0:
            return 19-(move)
        else:
             return 0

    def action2move(self,action):
        if action>=0 and action<20:
            return action + 1
        elif action>=20 and action<40:
            return 19-action
        else:
            return 0
    '''
    '''def move2action(self,move):
        if move>=1 and move<11:
            return move-1
        elif move>=-10 and move<0:
            return 9-(move)
        else:
             return 0'''

    def action2move(self,action,direction):
        if action>=0 and action<20:
            if direction:
                return action + 1
            else:
                return -(action + 1)
        else:
            if action == CLOCKWISE:
                return 11118
            elif action == ANTICLOCKWISE:
                return 81111
            elif action == KEEP_IMAGE:
                return 88888
            else:
                return -10000000000

    def calculate_position(self,init_state,steps):
        n_pos = init_state + steps
        if n_pos>(self.n_positions-1):
            n_pos -= self.n_positions
        elif n_pos<0:
            n_pos += self.n_positions
        return n_pos

    #gets the (shortest )distance between two positions (angles) in the circle
    def get_distance_two_positions(self,angle1,angle2):
        dist = angle2-angle1
        if dist > (self.n_positions/2):
            dist -= self.n_positions
        elif dist <= -(self.n_positions/2):
            dist += self.n_positions
        return dist


    def get_area_section_n(self, current_position):
        return int(current_position/(self.n_positions/self.n_zones))


    def set_setpoint(self, setpoint):
        self.setpoint = setpoint


    #gets idx ( and distance )of closest kept image (from current postition)
    def get_closest_idx(self,curr_pos,direction):
        if direction == 'right':
            st = 1
        else: #'left'
            st = -1

        mov_idx = curr_pos
        step_cnt = 0
        #move position until find a kept image
        while self.kept_im_map[mov_idx]!=1:
            mov_idx+= st
            mov_idx = mov_idx % self.n_positions
            step_cnt+=1
            if step_cnt > (self.n_positions/2):
                #max distance  is 180
                step_cnt = 100000000 #big number in order to be discarded when compared with another distance
                break
        return mov_idx, step_cnt


    #gets idx ( and distance )of closest kept image (from current postition)
    def get_closest_img(self,curr_pos):
       #get closest to right
       ridx, rdist = self.get_closest_idx(curr_pos,'right')
       #get closest to left
       lidx, ldist = self.get_closest_idx(curr_pos,'left')

       if rdist <= ldist:
           return ridx,rdist
       else:
           return lidx,-ldist


    def load_extrinsics(self,path):
        ext = []
        ext_files = glob.glob(os.path.join(path, '*.json'))
        assert len(ext_files) != 0,"json list is empty."
        for i in sorted(ext_files):                                                                                                                                  
            ext.append(json.load(open(i)))                                                                             
        return ext 
    
    def load_images(self,path,im_type):                                                                                                                                         
        imgs = []
        img_files = glob.glob(os.path.join(path, '*.'+im_type)) 
        assert len(img_files) != 0,"Image list is empty."
        for i in sorted(img_files):                                                                                                                                     
            img = Image.open(i)                                                                                                                                      
            imgs.append(img.copy())                                                                                                                                     
            img.close()                                                
        return imgs

    def set_sc(self,bbox):
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']
        z_min, z_max = bbox['z']

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1

        self.origin = np.array([x_min, y_min, z_min])
        self.sc = Backprojection([nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)

    def carve(self,idx):
        self.space_carve(self.rwd_images[idx], self.extrinsics[idx])
        
    def space_carve(self, im, rt):
        mask = get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation): mask = binary_dilation(mask)    
        self.sc.process_view(self.intrinsics, rot, tvec, mask)
        
    def dist_to_gt(self):
        vol = self.sc.values().copy()
        vol = vol.reshape(self.sc.shape)
        pcd=vol2pcd_exp(vol, self.origin, self.voxel_size, level_set_value=0) 
        pcd_p = np.asarray(pcd.points)
        cd=chamfer_d(self.gt_points , pcd_p)
        return cd 
            


