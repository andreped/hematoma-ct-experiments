import numpy as np
import subprocess as sp
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pytictoc import TicToc
import multiprocessing as mp
import dicom2nifti as dn
import dicom2nifti.settings as settings


### CONVERT GT PNG -> TIF ###

def func(loc):

	ids = loc.split("/")[6].split(" ")[0].split("CQ500CT")[1] + "_folder"
	out2 = new_path + ids + ".nii"
	out_dir = "/home/andrep/workspace/hematoma/NIFTI/" + ids

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# DICOM -> NIFTI
	dn.convert_directory(loc, out_dir, compression=False)

	# rename file in produced directory
	for file in os.listdir(out_dir):
		#print(file)
		out = "/home/andrep/workspace/hematoma/NIFTI/" + ids.split("_")[0] + "." + \
			".".join(file.split('.')[1:])
		os.rename(out_dir + "/" + file, out)
		os.rmdir(out_dir)
		sp.check_call(["gzip", out])


# BUGFIX: https://github.com/icometrix/dicom2nifti/issues/11
# sudo apt-get install libgdcm-tools   <- requires sudo...

if __name__ == "__main__":

	# disable gantry tilted GT validation
	settings.disable_validate_orthogonal()

	# disable slice increment validation
	settings.disable_validate_slice_increment()

	# initialize timer
	t = TicToc()
	t.tic()  # start

	loc = "/home/andrep/workspace/hematoma/"
	dicom_locs = []

	# choose image to convert
	path = "/home/andrep/workspace/hematoma/CQ500_alle/"
	new_path = "/home/andrep/workspace/hematoma/NIFTI/"

	if not os.path.exists(new_path):
	    os.makedirs(new_path)

	# paths in list
	locs = os.listdir(path)

	# remove all patients that ends with .zip, .txt, .csv
	tmp = []
	for p in locs:
		if not (p.endswith(".zip") or p.endswith(".txt") or p.endswith(".csv") or p.startswith(".")):
			tmp.append(p)

	locs = tmp.copy()

	files = []
	locs_dicom = []
	counter = 0
	for d1 in locs:
		if not d1.endswith('zip'):
			p1 = path + d1
			for d2 in os.listdir(p1):
				p2 = p1 + "/" + d2

				curr = ""
				flag = False
				longest_curr = 0
				for d4 in os.listdir(p2):
					p4 = p2 + "/" + d4
					tmp = len(os.listdir(p4))
					if tmp > longest_curr:
						longest_curr = tmp
						curr = p4
				if not curr == "":
					locs_dicom.append(curr)
					flag = True
				counter += 1
				if flag == True:
					files.append(counter)

				#for d3 in os.listdir(p2):
				#	p3 = p2 + "/" + d3
					#locs_dicom.append(p3)
					#exit()


					'''
					curr = ""
					flag = False
					longest_curr = 0
					for d4 in os.listdir(p3):
						p4 = p3 + "/" + d4
						print(p1)
						print(p2)
						print(p3)
						print(p4)
						tmp = len(os.listdir(p4))
						if tmp > longest_curr:
							longest_curr = tmp
							curr = p4
					if not curr == "":
						locs_dicom.append(curr)
						flag = True
					counter += 1
					if flag == True:
						files.append(counter)
					'''

	#locs_dicom = np.unique(locs_dicom)

	# run processes in parallel
	proc_num = 16 #16
	p = mp.Pool(proc_num)
	num_tasks = len(locs_dicom)
	r = list(tqdm(p.imap(func, locs_dicom), "WSI", total=num_tasks)) #list(tqdm(p.imap(func,gts),total=num_tasks))
	p.close()
	p.join()

	t.toc()
