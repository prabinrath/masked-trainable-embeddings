import cv2
import os
import glob
import pickle
import re

DATA_DIR = "/home/local/ASUAD/prath4/Repos/latent-actions/dataset/sim_open_close"
task_filter_map = {
        "box_open": re.compile(r"forward_[\d]*1.pickle"),
        "box_close": re.compile(r"backward_[\d]*1.pickle"),
        "door_open": re.compile(r"forward_[\d]*2.pickle"),
        "door_close": re.compile(r"backward_[\d]*2.pickle"),
    }
task_filter_key = task_filter_map['door_close']

for file_name in glob.glob(os.path.join(DATA_DIR, "*.pickle")):
    if task_filter_key.search(file_name):
        with open(file_name, "rb") as file:
            print(file_name)
            demo = pickle.load(file)
            # first_img = cv2.cvtColor(demo['front_rgb'][0], cv2.COLOR_RGB2BGR)
            # last_img = cv2.cvtColor(demo['front_rgb'][-1], cv2.COLOR_RGB2BGR)
            # cv2.imshow('first_img', first_img)
            # cv2.imshow('last_img', last_img)
            # cv2.waitKey(1000)
            for i in range(len(demo['right_shoulder_rgb'])):
                front_rgb = cv2.cvtColor(demo['right_shoulder_rgb'][i], cv2.COLOR_RGB2BGR)
                cv2.imshow('right_shoulder_rgb', front_rgb)
                cv2.waitKey(5)
