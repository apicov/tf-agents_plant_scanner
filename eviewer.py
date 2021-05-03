#import gym
import numpy as np
from collections import defaultdict, deque
import random
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json

import imp
import importlib.util
#spec = importlib.util.spec_from_file_location("romi_env_v6_4", "/home/pico/romi/scan_1440/romi_env_v6_4.py")


#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

import test_tools as nntools
imp.reload(nntools)

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
#sess = tf.Session(config=config) 

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print(tf.test.is_built_with_cuda())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import pygame
from pygame.locals import *
from pygame import gfxdraw
import sys
import numpy as np


def read_model(path,number):
    mname = 'model'+ str(number).zfill(3)
    # load json and create model
    json_file = open(path + mname + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path + mname + '.h5')
    return model

def read_model_n(path,name):
    # load json and create model
    json_file = open(path + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path + name + '.h5')
    return model



def get_circle_points(center,radius,npoints):
    angleStep = (2*np.pi)/npoints
    angles = np.arange(0,2*np.pi,angleStep)
    
    points = []
    #get points
    for i in angles:
        point = np.array([int(radius*np.cos(i)),int(radius*np.sin(i))])
        #translate point according to circle center
        point += center
        points.append(point)
    
    return points


def get_images(images_path):
    files = sorted([f for f in listdir(images_path) if isfile(join(images_path, f))])
    imgs = []
    for i in files:
        img = pygame.image.load(images_path+i)
        img = pygame.transform.scale(img, (640, 358))
        img = pygame.transform.flip(img, False, True)
        imgs.append(img)
    return imgs



BROWN = (139,69,19)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)


class policySim():
    def __init__(self,imgs):
        self.imgs = imgs
        pygame.display.set_caption('cnc simulator')
        size = 1024
        self.width = size
        self.height = size
        self.background = pygame.Surface((self.width,self.height))
        self.background.fill(BROWN)
        self.npoints = 1440
        self.radius = int((size*.9)/2.0)
        self.cam_pos = get_circle_points((int(self.width/2),int(self.height/2)),self.radius,self.npoints)
        pygame.draw.circle(self.background, BLACK ,(int(self.width/2),int(self.height/2)), self.radius,4)

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

    def reset(self):
        self.background = pygame.Surface((self.width,self.height))
        self.background.fill(BROWN)
        pygame.draw.circle(self.background, BLACK ,(int(self.width/2),int(self.height/2)), self.radius,4)
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

    def move(self,position):
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.imgs[position], (192, 323))
        pygame.draw.circle(self.screen, RED , self.cam_pos[position], 15)
        pygame.display.update()

    def save(self,position):
        #gfxdraw.pixel(self.background, self.cam_pos[position][0], self.cam_pos[position][1], GREEN)
        pygame.draw.circle(self.background, GREEN , self.cam_pos[position], 1)
        pygame.draw.circle(self.screen, GREEN , self.cam_pos[position], 15)
        pygame.display.update()


pygame.init()


#spec = importlib.util.spec_from_file_location("romi_env", "/home/pico/romi/scan_1440_2/models/idx22_50k/romi_env_v9_1440.py")
spec = importlib.util.spec_from_file_location("romi_env", "/home/pico/romi/scan_1440_2/romi_env_v9_1440.py")
romi_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(romi_env)
#mi = '/home/pico/romi/scan_1440/cosa/v01/'  
mi = '/home/pico/romi/scan_1440_2/real_plant002/v01/'
m_idxs = [mi+'inliers_ratios.npy']
setpoint = 0.03
envs = [romi_env.RomiEnv(env,setpoint) for env in m_idxs]

imgs = get_images(mi+'imgs/')

#mpath = '/home/pico/romi/scan_1440/models/6_4_10000/'
mpath = '/home/pico/romi/scan_1440_2/models/idx03/'
#model = read_model(mpath,0)#0
model = read_model_n(mpath,'idx03')#0

img_filename = 'realplant_002_v01.txt'


sim = policySim(imgs)
delay = 50

time_steps=1
#scaler =  nntools.MinMaxScaler(scale=(-1,1),mins=[0,-200,0,0]*time_steps,maxs=[100,200,36,20]*time_steps)
scaler =  nntools.MinMaxScaler(scale=(-1,1),mins=[0,-200,0,0,-1.0,-20,0]*time_steps,maxs=[1.0,200,18,10,1.0,20,1]*time_steps)

env = envs[0]
state = env.reset(0)
print(env.current_position,state)

sim.move(env.current_position)
sim.save(env.current_position)
pygame.time.delay(delay)

for i in range(5):
    pygame.time.delay(1000)
    print(i)

last_position = env.current_position
cont =0
obs = deque(maxlen=time_steps)

for i in range(time_steps-1):
    obs.append(state)

for i in range(3000):
    obs.append(state)
    flatten_state =  [item for sublist in obs for item in sublist]
    
    cont +=1
    action = np.argmax(  model.predict(scaler.transform(np.array([flatten_state])))[0]  )
    next_state,reward,done,info = env.step(action)
    
    print(cont,':',last_position,state,env.action2move(action,env.dir))
    if action == 22:
        sim.save(env.current_position)
    else:
        sim.move(env.current_position)
    pygame.time.delay(delay)
    
    state = next_state
    last_position = env.current_position
    if done is True:
        print('salio',cont)
        break
print(env.kept_images)
print(sorted(env.kept_images))
print('steps',cont)
print('total reward: ',env.total_reward)
print('n images: ',len(env.kept_images))

#np.savetxt(mpath+img_filename, env.kept_images, fmt='%d')


pygame.event.clear()
while True:
    event = pygame.event.wait()
    if event.type == QUIT:
        pygame.quit()
        sys.exit()
