import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import pandas as pd
import multiprocessing as mp
from datetime import date

import warnings

def func(path):
	path = str(path)
	#print(path)
	gt_path = gt_loc + path + "/" + path + "-label" + ".nrrd"
	data_path = data_loc + path + ".nii.gz"

	pat_id = path

	# make folder if not already exists
	if not os.path.isdir(end_path + pat_id):
		os.makedirs(end_path + pat_id)

	itkimage = sitk.ReadImage(data_path)
	data = sitk.GetArrayFromImage(itkimage).astype(np.float32)

	itkimage = sitk.ReadImage(gt_path)
	gt = sitk.GetArrayFromImage(itkimage).astype(np.float32)

	# fix orientation
	data = np.rot90(data, k=2, axes=(1, 2))
	data = np.flip(data, axis=2)
	gt = np.rot90(gt, k=2, axes=(1, 2))
	gt = np.flip(gt, axis=2)

	# HU-clip (# limits = (0, 100) # <- slicer config)
	#limits = (0, 140)
	 #(-230, 230)
	data[data < limits[0]] = limits[0]
	data[data > limits[1]] = limits[1]

	# scale -> [0, 1]
	#data = data - np.amin(data)
	#data = data / np.amax(data)

	# get resolution information
	img_size = out_dim[1:]
	tmp1 = itkimage.GetSpacing()
	tmp2 = itkimage.GetSize()
	res_factor = tmp1[2] / tmp1[0]
	res = np.ones(3)
	res[0] = res_factor
	res[1] = img_size[0] / tmp2[0]
	res[2] = img_size[1] / tmp2[1]

	# interpolate, (Nx512x512, with slice thickness 1)
	data = zoom(data, zoom=res, order=1)
	gt = zoom(gt.astype(np.float32), zoom=res, order=1)
	gt = (gt > th).astype(int)

	# scale data -> [0, 1]
	data = data - np.amin(data)
	data = data / np.amax(data)

	# split data into overlapping chunks
	cnt = 0
	# for i in range(int(np.ceil(data.shape[0] / out_dim[0]))):
	for i in range(int(np.ceil(data.shape[0] / stride))):
		# if not np.count_nonzero(gt[i * stride:i * stride + out_dim[0]]) == 0: # pos only
		if True:  # all
			data_out = np.zeros(out_dim, dtype=np.float32)
			tmp = data[i * stride:i * stride + out_dim[0]]
			data_out[:tmp.shape[0]] = tmp

			gt_out = np.zeros(out_dim, dtype=np.float32)
			tmp = gt[i * stride:i * stride + out_dim[0]]
			gt_out[:tmp.shape[0]] = tmp

			# fix arrays
			data_out = np.expand_dims(data_out, axis=0)
			gt_out = np.expand_dims(gt_out, axis=0)
			data_out = np.expand_dims(data_out, axis=-1)
			tmp = np.zeros(gt_out.shape + (2,), dtype=np.float32)
			tmp[..., 0] = 1 - gt_out
			tmp[..., 1] = gt_out
			gt_out = tmp.copy()
			del tmp

			if not np.count_nonzero(gt[i * stride:i * stride + out_dim[0]]) == 0:
				flag = 1
			else:
				flag = 0

			f = h5py.File(end_path + pat_id + "/" + str(cnt) + "_" + str(flag) + ".h5", "w")
			f.create_dataset("input", data=data_out, compression="gzip", compression_opts=4)
			f.create_dataset("output", data=gt_out.astype(np.uint8), compression="gzip", compression_opts=4)
			f.close()

			cnt += 1


if __name__ == '__main__':

	out_dim = (64, 256, 256)
	limits = (0, 140)
	#limits = (-230, 230)
	th = 0.5
	stride = 4

	warnings.filterwarnings('ignore', '.*output shape of zoom.*')
	data_loc = "/home/andrep/workspace/hematoma/data/"
	gt_loc = "/home/andrep/workspace/hematoma/gt/"
	#end_path = "/home/andrep/hematoma/datasets/data_16_512_all_split/"
	#end_path = "/home/andrep/workspace/hematoma/datasets/data_16_512_pos_only_stride_4/"
	end_path = "/home/andrep/workspace/hematoma/datasets/data_16_512_all_stride_" + str(stride) + "_" + "out_dim_"\
			   + str(out_dim).replace(" ", "") + "_limits_" + str(limits).replace(" ", "") + "/"

	if not os.path.isdir(end_path):
		os.makedirs(end_path)

	tmp = os.listdir(gt_loc)
	remove = ["Bleeding", "Liste"]
	locs = []
	for l in tmp:
		if not((remove[0] in l) or (remove[1] in l)):
			locs.append(l)

	# filter based on old/new bleeding from excel file
	excel = pd.read_excel(gt_loc + "Liste hjerneblødninger.xlsx")
	ids = excel["ID"][(excel["Sendt til SINTEF?"] == "Ja") & (excel["Ny  (lys)"] == 1.0) & (excel["Detection"] != 1.0)]
	ids = np.array(ids).flatten()

	for l in locs:
		if not int(l) in ids:
			locs.remove(l)

	locs = np.array(locs)
	locs = locs.astype(np.int64)
	locs = np.sort(locs)

	proc_num = 16
	p = mp.Pool(proc_num)
	num_tasks = len(locs)
	r = list(tqdm(p.imap(func, locs), "WSI", total=num_tasks))  # list(tqdm(p.imap(func,gts),total=num_tasks))
	p.close()
	p.join()
