import os
import cv2
import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt

from scipy.io import loadmat


def modify(phase):

	if phase == "Train":
		path_to_image = "data/consep/CoNSeP/Train/Images"
		path_to_overlay = "data/consep/CoNSeP/Train/Overlay"
		path_to_labels = "data/consep/CoNSeP/Train/Labels"
		out_path = "data/full_full_data/Train_out"
	else:
		path_to_image = "data/consep/CoNSeP/Test/Images"
		path_to_overlay = "data/consep/CoNSeP/Test/Overlay"
		path_to_labels = "data/consep/CoNSeP/Test/Labels"
		out_path = "data/full_full_data/Test_out"


	image_names = sorted(os.listdir(path_to_image))
	overlay_names = sorted(os.listdir(path_to_overlay))
	label_names = sorted(os.listdir(path_to_labels))

	for img_name, ovl_name, label_name in zip(image_names, overlay_names, label_names):
		img_array = cv2.imread(os.path.join(path_to_image, img_name))
		img_color = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
		img_gray = cv2.imread(os.path.join(path_to_image, img_name), 0)
		img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)

		ovl_gray = cv2.imread(os.path.join(path_to_overlay, ovl_name), 0)
		ovl_gray = cv2.cvtColor(ovl_gray, cv2.COLOR_BGR2RGB)
		#pixel-level label
		data = loadmat(os.path.join(path_to_labels,label_name))['type_map']

		print(f"Cropping image {img_name}...")
		right_side = ovl_gray-img_gray

		for i in range(4):
			for j in range(4):
				if i != 3 and j !=3:
					original = img_color[i*256:i*256+256, j*256:j*256+256]
					label = right_side[i*256:i*256+256, j*256:j*256+256]
					pxl_label = data[i * 256:i * 256 + 256, j * 256:j * 256 + 256]

				elif i == 3 and j !=3:
					original = img_color[1000-256:1000, j*256:j*256+256]
					label = right_side[1000-256:1000, j*256:j*256+256]
					pxl_label = data[1000 - 256:1000, j * 256:j * 256 + 256]
		            
				elif i !=3 and j ==3:
					original = img_color[i*256:i*256+256, 1000-256:1000]
					label = right_side[i*256:i*256+256, 1000-256:1000]
					pxl_label = data[i * 256:i * 256 + 256, 1000 - 256:1000]
		            
				else:
					original = img_color[1000-256:1000, 1000-256:1000]
					label = right_side[1000-256:1000, 1000-256:1000]
					pxl_label = data[1000 - 256:1000, 1000 - 256:1000]

				img_folder = img_name.split('.')[0]
				out_path_folder = os.path.join(out_path, img_folder+"_folder"+str(i)+str(j))

				if not os.path.exists(out_path_folder):
					os.makedirs(out_path_folder)

				cv2.imwrite(os.path.join(out_path_folder, 'image.png'), original)
				cv2.imwrite(os.path.join(out_path_folder, 'mask.png'), label)
				np.save(os.path.join(out_path_folder,'label.npy'), pxl_label)


if __name__ == "__main__":
	modify("Train")
	modify("Test")