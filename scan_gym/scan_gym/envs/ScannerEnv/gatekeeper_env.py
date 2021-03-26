import numpy as np
#from PIL import Image
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

#import glob
#import re
'''
MAP =  ["--.G",
        "---.",
        "-++-",
        "----"]
'''
'''MAP =  ["--.G",
        "----",
        "+++-",
        "--+-"]'''

MAP =  ["---G",
        "----",
        "----",
        "----"]

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

GATEKEEPER_TYPES = ['L','D','R','U'] #idx is equivalent to action required by the gatekeeper type ('L'=0=LEFT,'R'=2=RIGHT)
GATEKEEPER_LOCATIONS = [(0,2),(1,3)]#[(1,3),(0,2)]#(0,2),,(1,3)
BLOCK_LOCATIONS = [(2,1),(2,2)]
GOAL_LOCATION = (0,3)

CELL_IMAGES = {"-":np.array([[255,255],[255,255]]), "L":np.array([[0,255],[0,255]]), "D":np.array([[255,255],[0,0]]), "R":np.array([[255,0],[255,0]]), "U":np.array([[0,0],[255,255]]), "G":np.array([[0,0],[0,0]])   }

class GatekeeperEnv(gym.Env):
    def __init__(self):#, params):
        super(GatekeeperEnv, self).__init__()

        self.max_steps = 100
        self.step_count = 0

        self.action_space = spaces.Discrete(4)
        self.position_ob_space  =  spaces.Discrete(16)
        self.im_ob_space = spaces.Box(low=0, high=255, shape=(2,2))#, dtype=np.uint8)
        self.observation_space = spaces.Tuple((self.im_ob_space, self.position_ob_space))

        self.reset()

        
    
    def reset(self,init_pos=None):
        self.done = False
        self.step_count = 0
        self.map = np.asarray(MAP, dtype='c')
        self.insert_rnd_gatekeepers() #select randomly two gatekeepers from 4 possible types
        self.insert_blocks(BLOCK_LOCATIONS)
        if init_pos == None:
            #self.curr_loc = self.rnd_start_position(self.map.shape)
            self.curr_loc =(3,3) # (3,1) #starting position
        else:
            self.curr_loc = init_pos
        #self.curr_loc =(3,3) # (3,1) #starting position
        self.curr_img = CELL_IMAGES[self.map[self.curr_loc[0],self.curr_loc[1]].decode('UTF-8')]
        return (self.curr_img, self.vec2n(self.curr_loc))
      
    def step(self, action):
        if self.curr_loc in  GATEKEEPER_LOCATIONS:
            #if agent is located in gatekeeper, check if the executed action is the one asked
            if self.is_right_action(self.map[self.curr_loc[0],self.curr_loc[1]].decode('UTF-8'),action):
                if self.curr_loc == (1,3):
                    reward = 50
                else:
                    reward=50 #goal achieved
                self.curr_loc= GOAL_LOCATION
                self.curr_img = CELL_IMAGES['G']
            else: #wrong password, agent loses
                reward= -50
                self.curr_img = CELL_IMAGES[self.map[self.curr_loc[0],self.curr_loc[1]].decode('UTF-8')]
            self.done = True
        
        else:
            reward = -1
            new_loc = self.inc(self.curr_loc[0], self.curr_loc[1], action)
            #if agent moves over a block, return to previous position
            if self.map[new_loc[0],new_loc[1]] != b'+':
                self.curr_loc = new_loc
            self.curr_img = CELL_IMAGES[self.map[self.curr_loc[0],self.curr_loc[1]].decode('UTF-8')]
            #self.curr_loc = new_loc

            #if self.curr_loc == (2,3):
            #    print('pipi')

            if self.curr_loc == GOAL_LOCATION:
                reward = 10
                self.done = True

        #print(self.curr_loc )
        self.step_count+= 1
        if self.step_count >= self.max_steps:
            self.done =True

        return (self.curr_img, self.vec2n(self.curr_loc)), reward, self.done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return 0


    def insert_rnd_gatekeepers(self):
        gk = random.sample(GATEKEEPER_TYPES, len(GATEKEEPER_LOCATIONS))
        for l,k in zip(GATEKEEPER_LOCATIONS,gk):
            self.map[l[0],l[1]] = k

    def insert_blocks(self,blocks):
        for b in BLOCK_LOCATIONS:
            self.map[b[0],b[1]] = b'+'

    def rnd_start_position(self,map_size):
        #select initial random position, block locations are not allowed
        sp = (random.randint(0, map_size[0]-1), random.randint(0, map_size[1]-1)  )
        while True:
            if sp not in BLOCK_LOCATIONS:
                break
            else:
                sp = (random.randint(0, map_size[0]-1), random.randint(0, map_size[1]-1)  )
        return sp


    def vec2n(self,vec):
        return vec[0]*self.map.shape[1] + vec[1]

    def inc(self,row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.map.shape[0] - 1)
        elif a == RIGHT:
            col = min(col + 1, self.map.shape[1] - 1)
        elif a == UP:
            row = max(row - 1, 0) 
        return row, col

    def is_right_action(self,pwd,action):
        if GATEKEEPER_TYPES[action] == pwd:
            return True
        else:
            return False
