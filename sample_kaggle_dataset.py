# USAGE
# python sample_kaggle_dataset.py --kaggle chest_xray --output dataset/normal

"""
--kaggle
/Volumes/MacBackup/kaggle-chest-x-ray-images/chest_xray/train/NORMAL
--output
./dataset/0318/normal
--sample
102
"""

# import the necessary packages
from imutils import paths
import argparse
import random
import shutil
import os
from pathlib import Path

def create_kaggle_dataset(kaggle_dataset_dir: str, output_dir: str, number_of_images: int):
	file_count = 0

	# make dir if output does not exist
	Path(output_dir).mkdir(exist_ok=True, parents=True)

	# grab all training image paths from the Kaggle X-ray dataset
	# basePath = os.path.sep.join([args["kaggle"], "train", "NORMAL"])
	imagePaths = list(paths.list_images(kaggle_dataset_dir))

	# randomly sample the image paths
	random.seed(42)
	random.shuffle(imagePaths)
	imagePaths = imagePaths[:number_of_images]

	output_label = output_dir.split(os.path.sep)[-1]
	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the filename from the image path and then construct the
		# path to the copied image file
		filename = imagePath.split(os.path.sep)[-1]
		filename = f"kaggle_{output_label}_{file_count}.{filename.split('.')[-1]}"

		outputPath = os.path.sep.join([output_dir, filename])

		file_count += 1
		# copy the image
		shutil.copy2(imagePath, outputPath)


	return file_count

if __name__ == '__main__':
	print("Process Kaggle Chest X-Ray Dataset")
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-k", "--kaggle", required=True,
		help="path to base directory of Kaggle X-ray dataset")
	ap.add_argument("-o", "--output", required=True,
		help="path to directory where 'normal' images will be stored")
	ap.add_argument("-s", "--sample", type=int, default=25,
		help="# of samples to pull from Kaggle dataset")
	args = vars(ap.parse_args())

	kaggle_dir = args["kaggle"]
	output_dir = args['output']
	count = args['sample']

	print(kaggle_dir, output_dir, count)
	file_count = create_kaggle_dataset(kaggle_dir, output_dir, count)
	print(f"{file_count} Kaggle images used from directory: {kaggle_dir} placed in output directory: {output_dir}")
