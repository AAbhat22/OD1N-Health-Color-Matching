# -*- coding: utf-8 -*-

from collections import Counter

#  Color detection
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

train_img = cv2.imread('./Images/reference.jpg')  # set path to reference image.
test_img = cv2.imread('./Images/testing.png')  # set path to testing image.

gray_image = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


"""Using ML model K mean clustering to find out the RGB range of colors in the image by creating 63 clusters which are number of colors in the referencing image."""

modified_image = cv2.resize(train_img, (600, 400), interpolation=cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
clf = KMeans(n_clusters=63)
labels = clf.fit_predict(modified_image)
counts = Counter(labels)
center_colors = clf.cluster_centers_
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

for i in range(len(rgb_colors)):
    rgb_colors[i] = rgb_colors[i].astype(int)
print(rgb_colors)

img = test_img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Total numbers of parts in the image", len(contours))
temp = list(contours)

new_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 100 and h > 30:
        new_contours.append(contour)

new_contours.pop(0)

"""Calculating the Euclidian distance to find out the nearest match."""


def find_match(avg_color):
    smallest_dis = distance.euclidean(rgb_colors[0], avg_color)
    smallest_dis_color = []
    for color in rgb_colors:
        if distance.euclidean(color, avg_color) < smallest_dis:
            smallest_dis = distance.euclidean(color, avg_color)
            smallest_dis_color = color

    # print("neareest match " ,smallest_dis_color)
    return smallest_dis_color


for cnt in new_contours:
    predicted_image = np.zeros((100, 130, 3), np.uint8)
    if cv2.contourArea(cnt) > 800:  # filter small contours
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        current_cnt_img = img[y:y + h, x:x + w]
        bgr = np.array(cv2.mean(current_cnt_img))[:3].astype(np.uint8)
        rgb = [bgr[2], bgr[1], bgr[0]]

        predicted_color = find_match(bgr)
        predicted_image[:] = predicted_color
        print("part of testing image  with real color ", bgr)
        cv2.imshow("Current part", current_cnt_img)
        cv2.waitKey(1)
        print("image with predcited color :", predicted_color)
        cv2.imshow("prediction", predicted_image)
        cv2.waitKey(1)
