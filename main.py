from re import I
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops


def extract_features(image):
    features = []
    # [Next, Previous, First_child, Parent]
    _, hierachy = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # lakes = 0 if len(hierachy[0]) == 1 else 1
    # features.append(lakes)
    ext_cnt = 0
    int_cnt = 0
    for i in range(len(hierachy[0])):
        if hierachy[0][i][-1] == -1:
            ext_cnt += 1
        elif hierachy[0][i][-1] == 0:
            int_cnt += 1

    features.extend([ext_cnt, int_cnt])
    labeled = label(image)
    region = regionprops(labeled)[0]
    filling_factor = region.area / region.bbox_area
    features.append(filling_factor)
    centroid = np.array(region.local_centroid) / np.array(region.image.shape)
    features.extend(centroid)
    features.append(region.eccentricity)
    return features


train_dir = Path("./data/out/") / "train"
train_data = defaultdict(list)

for path in sorted(train_dir.glob("*")):
    if path.is_dir():
        for img_path in path.glob("*.png"):
            symbol = path.name[-1]
            gray = cv2.imread(str(img_path), 0)
            binary = gray.copy()
            binary[binary > 0] = 1
            train_data[symbol].append(binary)

features_array = []
responses = []
for i, symbol in enumerate(train_data):
    # print(i)
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))

features_array = np.array(features_array, dtype="f4")
responses = np.array(responses)

knn = cv2.ml.KNearest_create()
knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)

test_symbol = extract_features(train_data["p"][0])
test_symbol = np.array(test_symbol, dtype="f4").reshape(1, 6)
ret, results, neighbours, dist = knn.findNearest(test_symbol, 3)

print(chr(int(ret)), results, neighbours, dist)

# print(features_array.shape)
# print(responses.shape)
# letter = input()
# print(list(train_data))
# # print(extract_features(train_data[letter][0]))
# plt.imshow(train_data[letter][0])
# plt.figtext(0.5, 0.01, extract_features(train_data[letter][0]), ha="center", fontsize=8, bbox={
#             "facecolor": "orange", "alpha": 0.5, "pad": 5})
# plt.show()
