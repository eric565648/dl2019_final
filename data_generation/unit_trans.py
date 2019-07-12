#!/usr/bin/env python

import csv
import os
import math

set = "set03"

train_label_csv = set+'_train_label.csv'
test_label_csv = set+'_test_label.csv'

new_train_label_csv = set+'_train_label_new.csv'
new_test_label_csv = set+'_test_label_new.csv'

if os.path.exists(new_train_label_csv):
    os.remove(new_train_label_csv)
if os.path.exists(new_test_label_csv):
    os.remove(new_test_label_csv)

with open(train_label_csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        this_row = row[0].split(',')
        d = float(this_row[0]) / 100 + 0.35
        phi = (math.radians(float(this_row[1])) + math.pi/2) / math.pi
        img_gt = [d, phi]
        with open(new_train_label_csv, 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([img_gt])

with open(test_label_csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        this_row = row[0].split(',')
        d = float(this_row[0]) / 100 + 0.35
        phi = (math.radians(float(this_row[1])) + math.pi/2) / math.pi
        img_gt = [d, phi]
        with open(new_test_label_csv, 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([img_gt])
