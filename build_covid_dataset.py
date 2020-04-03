# USAGE
# python build_covid_dataset.py --covid covid-chestxray-dataset --output dataset/covid
# --covid /Volumes/MacBackup/covid-chestxray-dataset --output ./dataset/0318/covid
"""
Largely the same as the one from PyImageSearch Blog but refactored to:

* Allow function to be called in Jupyter Notebook
* Automatically add output path if it does not exist
* Return the file count


"""
# import the necessary packages
import pandas as pd
import shutil
import os
from pathlib import Path


def create_covid_dataset(covid_dataset_dir: str, output_dir: str):
    file_count = 0

    # make dir if output does not exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # construct the path to the metadata CSV file and load it
    csvPath = os.path.sep.join([covid_dataset_dir, "metadata.csv"])
    df = pd.read_csv(csvPath)

    # loop over the rows of the COVID-19 data frame
    for (i, row) in df.iterrows():
        # if (1) the current case is not COVID-19 or (2) this is not
        # a 'PA' view, then ignore the row
        if row["finding"] != "COVID-19" or row["view"] != "PA":
            continue

        # build the path to the input image file
        imagePath = os.path.sep.join([covid_dataset_dir, "images", row["filename"]])

        # if the input image file does not exist (there are some errors in
        # the COVID-19 metadeta file), ignore the row
        if not os.path.exists(imagePath):
            continue

        # extract the filename from the image path and then construct the
        # path to the copied image file
        # create a filename like:  covid19_{index}.{suffix}
        filename = row["filename"].split(os.path.sep)[-1]
        filename = f"covid19_{file_count}.{filename.split('.')[-1]}"
        outputPath = os.path.sep.join([output_dir, filename])

        file_count += 1
        # copy the image
        shutil.copy2(imagePath, outputPath)

    return file_count

if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--covid", required=False, default='/Volumes/MacBackup/covid-chestxray-dataset',
                    help="path to base directory for COVID-19 dataset")
    ap.add_argument("-o", "--output", required=False, default='./dataset/0402/covid',
                    help="path to directory where 'normal' images will be stored")
    args = vars(ap.parse_args())

    covid_dir = args["covid"]
    output_dir = args['output']

    file_count = create_covid_dataset(covid_dir, output_dir)
    print(f"{file_count} COVID-19 files added from dataset directory[{covid_dir}] to output directory[{output_dir}]")
