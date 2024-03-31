import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from tqdm import tqdm
from numpy.random import shuffle
from shutil import copyfile


#loc = "/hdd/hematoma/NIFTI/0.nii.gz"
path = "/hdd/hematoma/NIFTI/"

res_xy = []
res_z = []

names = []

for loc in tqdm(os.listdir(path)):

	names.append(loc)

	loc = path + loc

	reader = sitk.ImageFileReader()
	reader.SetFileName(loc)
	reader.ReadImageInformation()

	for k in reader.GetMetaDataKeys():
		v = reader.GetMetaData(k)
		#print(k, v)
	#print()


	#itkimage = sitk.ReadImage(loc)
	#data = sitk.GetArrayFromImage(itkimage)
	#im = ct_scan[0,:,:]
	'''
	fig, ax = plt.subplots()
	for i in range(data.shape[0]):
		ax.imshow(data[i], cmap="gray")
		plt.pause(0.01)
	plt.show()
	'''

	# get resolution information
	labs = ["pixdim[1]", "pixdim[2]", "pixdim[3]"]
	res = []
	for l in labs:
		res.append(float(reader.GetMetaData(l)))

	res_xy.append(res[0])
	res_z.append(res[2])

	#print()
	#print('XYZ resolution: ')
	#print(res)
	#print(data.shape)

	if res[2] > 10:
		print(loc)

res_xy = np.array(res_xy)
res_z = np.array(res_z)

th1 = 0.7
count1_xy = sum(res_xy <= th1)
count2_xy = sum(res_xy > th1)

th2 = 2.5
count1_z = sum(res_z <= th2)
count2_z = sum(res_z > th2)

th3 = 4
count3_z = sum(res_z <= th3)
count4_z = sum(res_z > th3)

print()
print(names)
names = np.array(names)
names = names[res_z <= 1]

shuffle(names)
names = names[:50]
print(names)


res_z = []

for loc in tqdm(names):

	loc = path + loc

	reader = sitk.ImageFileReader()
	reader.SetFileName(loc)
	reader.ReadImageInformation()

	for k in reader.GetMetaDataKeys():
		v = reader.GetMetaData(k)
		#print(k, v)
	#print()


	#itkimage = sitk.ReadImage(loc)
	#data = sitk.GetArrayFromImage(itkimage)
	#im = ct_scan[0,:,:]
	'''
	fig, ax = plt.subplots()
	for i in range(data.shape[0]):
		ax.imshow(data[i], cmap="gray")
		plt.pause(0.01)
	plt.show()
	'''

	# get resolution information
	labs = ["pixdim[1]", "pixdim[2]", "pixdim[3]"]
	res = []
	for l in labs:
		res.append(float(reader.GetMetaData(l)))

	res_z.append(res[2])

plt.hist(res_z)
plt.show()

new_path = "/hdd/hematoma/CQ500_set1_50/"
if not os.path.exists(new_path):
	os.makedirs(new_path)

for name in names:
	copyfile(path + name, new_path + name)


exit()


print('xy, ' + str(th1))
print(count1_xy, count2_xy)
print()
print('z-threshold: ' + str(th2))
print(count1_z, count2_z)
print()
print('z-threshold: ' + str(th3))
print(count3_z, count4_z)
print()


plt.figure()
plt.hist(res_xy, bins=30)
plt.title("res xy")

plt.figure()
plt.hist(res_z, bins=30)
plt.title("res z")
plt.show()
