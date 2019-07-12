#!/usr/bin/env python

"""
This scripts is the testing scripts to generate learning data
from gym-duckietown
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--max-pics', default=1000, type=int, help='Max images per tile')
parser.add_argument('--file-name', default='foo', help='the csv file name')
args = parser.parse_args()

print(args)
print(args.env_name)
print(args.env_name.find('Duckietown'))

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

# assert isinstance(env.unwrapped, Simulator)

# @env.unwrapped.window.event
# def on_key_press(symbol, modifiers):
#     global tile_id
#     """
#     This handler processes keyboard commands that
#     control the simulation
#     """
#
#     if symbol == key.BACKSPACE or symbol == key.SLASH:
#         generate_image(0)
#
#         return
#     elif symbol == key.ESCAPE:
#         env.close()
#         sys.exit(0)

def generate_image(id, it, save_path):

    print("Start coords: ",env.drivable_tiles[id]['coords'])
    i, j = env.drivable_tiles[id]['coords']
    env.user_tile_start = [i, j]
    obs = env.reset()
    env.render()

    lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    print("LP:",lp)

    img_gt = [lp.dist, lp.angle_rad]

    from PIL import Image
    im = Image.fromarray(obs)

    img_name = str(id)+"_"+str(i)+"_"+str(j)+"_"+str(it)

    im.save(save_path+img_name+".png")

    # tile_id += 1
    # if tile_id == len(env.drivable_tiles):
    #     tile_id = 0

    return img_name, img_gt;

# pyglet.app.run()

if not os.path.isdir(args.file_name):
    os.mkdir(args.file_name)
file_path = args.file_name + '/'

train_label_csv = args.file_name+'_train_label.csv'
train_img_csv = args.file_name+'_train_img.csv'
test_label_csv = args.file_name+'_test_label.csv'
test_img_csv = args.file_name+'_test_img.csv'
if os.path.exists(train_label_csv):
    os.remove(train_label_csv)
if os.path.exists(train_img_csv):
    os.remove(train_img_csv)
if os.path.exists(test_label_csv):
    os.remove(test_label_csv)
if os.path.exists(test_img_csv):
    os.remove(test_img_csv)

MAX_PICS = args.max_pics
for tile_id in range(len(env.drivable_tiles)):
    for it_gen in range(MAX_PICS):
        i_name, i_gt = generate_image(tile_id, it_gen, file_path)

        if it_gen%10 == 0:
            with open(test_img_csv, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow([i_name])
            with open(test_label_csv, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows([i_gt])
        else:
            with open(train_img_csv, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow([i_name])
            with open(train_label_csv, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows([i_gt])

env.close()
