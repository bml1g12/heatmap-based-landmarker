#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import time
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import cv2
import sys
from models.heatmapmodel import HeatMapLandmarker, \
    heatmap2coord, heatmap2topkheatmap, lmks2heatmap, loss_heatmap, heatmap2softmaxheatmap, \
    heatmap2sigmoidheatmap, mean_topk_activation
from datasets.dataLAPA106 import LAPA106DataSet
from torchvision import transforms
from retinaface import RetinaFace

THRESH_OCCLUDED = 0.5

# Transform
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def square_box(box, ori_shape):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w = max(x2 - x1, y2 - y1) * 1.2
    x1 = cx - w // 2
    y1 = cy - w // 2
    x2 = cx + w // 2
    y2 = cy + w // 2

    x1 = max(x1, 0)
    y1 = max(y1 + (y2 - y1) * 0, 0)
    x2 = min(x2 - (x1 - x1) * 0, ori_shape[1] - 1)
    y2 = min(y2, ori_shape[0] - 1)

    return [x1, y1, x2, y2]


def draw_landmarks(img, lmks, point_occluded, color=(0, 255, 0)):
    default_color = color
    for a, is_occluded in zip(lmks, point_occluded):
        if is_occluded:
            color = (0, 0, 255)

        else:
            color = default_color

        cv2.circle(img, (int(round(a[0])), int(round(a[1]))), 2, color, -1, lineType=cv2.LINE_AA)

    return img


def concat_gt_heatmap(heat):
    """
    \ Heat size : 106 x 64 x 64
    """
    # print(f'Shape Gt heatmap: {heat.shape}')
    heat = heat.numpy()
    heat = np.max(heat, axis=0)
    heat = heat * 255.0 / (np.max(heat))

    return heat


def apply_to_image(detector, model, img):
    # Get box detector and then make it square
    faces = detector.predict(img)
    if len(faces) != 0:
        box = [faces[0]['x1'], faces[0]['y1'], faces[0]['x2'], faces[0]['y2']]
        box = square_box(box, img.shape)
        box = list(map(int, box))
        x1, y1, x2, y2 = box
        # Inference lmks
        crop_face = img[y1:y2, x1:x2]
        crop_face = cv2.resize(crop_face, (256, 256))
        img_tensor = TRANSFORM(crop_face)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # 1x3x256x256
        heatmapPRED, lmks = model(img_tensor.to(DEVICE))
        scores = mean_topk_activation(heatmapPRED.to('cpu'), topk=3)[0]
        # scores now torch.Size([106, 1, 3])
        scores = scores.view(106, -1)
        # scores now torch.Size([106, 3])
        scores = torch.mean(scores, dim=-1)
        # score now torch.Size([106])
        point_occluded = scores < THRESH_OCCLUDED
        # point_occluded is a bool array of length 106
        lmks = lmks.cpu().detach().numpy()[0]  # 106x2
        # lmks is shape (106, 2)
        lmks = lmks / 256.0  # Scale into 0-1 coordination
        lmks[:, 0], lmks[:, 1] = lmks[:, 0] * (x2 - x1) + x1, \
                                 lmks[:, 1] * (y2 - y1) + y1
        img = draw_landmarks(img, lmks, point_occluded)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img


def load_keypoint_model(checkpoint_path):
    model = HeatMapLandmarker()
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['plfd_backbone'])
    model.to(DEVICE)
    model.eval()
    return model


def main():
    filepath = "Selfie.jpg"
    img = cv2.imread(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)
    if img is None:
        print(f"could not read image {filepath}, exiting")
        return
    face_detector = RetinaFace(quality="normal")
    keypoint_model = load_keypoint_model("./ckpt/epoch_80.pth.tar")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    img = apply_to_image(face_detector, keypoint_model, img)
    cv2.imshow("Image", img)
    k = cv2.waitKey(0)
    if k == ord("q"):
        return


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
