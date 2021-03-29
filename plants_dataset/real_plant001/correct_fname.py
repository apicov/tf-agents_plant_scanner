import os
from os import listdir
from os.path import isfile, join
import sys

imgs_path = sys.argv[1]
if imgs_path[-1] != '/':
    imgs_path += '/'

files = [f for f in listdir(imgs_path) if isfile(join(imgs_path, f))]

for f in files:
    sidx = f.find('_')
    num = int(f[:sidx])
    rest = f[sidx:]
    newname = str(num).zfill(4) + rest
    os.rename(imgs_path+f, imgs_path+newname)

print('listo!')
