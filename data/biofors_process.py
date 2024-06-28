import json
import os
import shutil

import cv2
import numpy

cls_json = "/home/hdd1/share/public_data/biofors/annotation_files/classification.json"
idd_json = "/home/hdd1/share/public_data/biofors/annotation_files/idd_gt.json"
cstd_json = "/home/hdd1/share/public_data/biofors/annotation_files/cstd_gt.json"
auth_dir = "/home/hdd1/share/public_data/bio_imd_2023/biofors/auth"
mask_dir = "/home/hdd1/share/public_data/bio_imd_2023/biofors/mask"
cstd_dir = "/home/hdd1/share/public_data/bio_imd_2023/biofors/cstd"
biofors_dir = "/home/hdd1/share/public_data/biofors"
new_dir = ""

cls_dict = {
    "Microscopy": "microscopy",
    "Macroscopy": "macroscopy",
    "Blot/Gel": "blot",
    "FACS": "FACS"
}

with open(cls_json, 'r') as f:
    cls = json.load(f)
with open(idd_json, 'r') as f:
    idd = json.load(f)
with open(cstd_json, 'r') as f:
    cstd = json.load(f)

for paper_id in cstd:
    for image_name in cstd[paper_id]:
        gt = cstd[paper_id][image_name]
        cls_name = cls[paper_id][image_name]
        # print(cls_name)
        cls_name = cls_dict[cls_name]
        source_dir = os.path.join(biofors_dir, paper_id, image_name)
        dest_dir = os.path.join(cstd_dir, cls_name, f"{paper_id}-{image_name}")
        print(source_dir, dest_dir)
        shutil.copy(source_dir, dest_dir)
        img = cv2.imread(source_dir)
        mask = numpy.zeros((img.shape[0], img.shape[1]))

        for points in gt:
            mask = cv2.rectangle(mask, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), 255, -1)
        cv2.imwrite(dest_dir.replace(cstd_dir, mask_dir), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

for paper_id in cls:
    for image_name in cls[paper_id]:
        if paper_id in cstd:
            continue
        if paper_id in idd:
            continue

        cls_name = cls[paper_id][image_name]
        cls_name = cls_dict[cls_name]
        source_dir = os.path.join(biofors_dir, paper_id, image_name)
        dest_dir = os.path.join(auth_dir, cls_name, f"{paper_id}-{image_name}")
        shutil.copy(source_dir, dest_dir)