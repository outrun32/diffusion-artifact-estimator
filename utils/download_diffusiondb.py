# Author: Marco Lustri 2022 - https://github.com/TheLustriVA
# MIT License

"""A script to make downloading the DiffusionDB dataset easier."""
from urllib.error import HTTPError
from urllib.request import urlretrieve
from alive_progress import alive_bar
from os.path import exists
import os

import shutil
import os
import time
import argparse

index = 3  # initiate main arguments as None
range_max = None
output = 'images/'
unzip = True
large = True



def download(index=1, range_index=0, output="", large=False):
    """
    Download a file from a URL and save it to a local file

    :param index: The index of the file to download, defaults to 1 (optional)
    :param range_index: The number of files to download. If you want to download
        all files, set this to the number of files you want to download,
        defaults to 0 (optional)
    :param output: The directory to download the files to :return: A list of
        files to unzip
    :param large: If downloading from DiffusionDB Large (14 million images)
        instead of DiffusionDB 2M (2 million images)
    """
    baseurl = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/"
    files_to_unzip = []

    if large:
        if index <= 10000:
            url = f"{baseurl}diffusiondb-large-part-1/part-{index:06}.zip"
        else:
            url = f"{baseurl}diffusiondb-large-part-2/part-{index:06}.zip"
    else:
        url = f"{baseurl}images/part-{index:06}.zip"

    if output != "":
        output = f"{output}/"

    if not exists(output):
        os.makedirs(output)

    if range_index == 0:
        print("Downloading file: ", url)
        file_path = f"{output}part-{index:06}.zip"
        try:
            urlretrieve(url, file_path)
        except HTTPError as e:
            print(f"Encountered an HTTPError downloading file: {url} - {e}")
        if unzip:
            unzip_file(file_path)
    else:
        # It's downloading the files numbered from index to range_index.
        with alive_bar(range_index - index, title="Downloading files") as bar:
            for idx in range(index, range_index):
                if large:
                    if idx <= 10000:
                        url = f"{baseurl}diffusiondb-large-part-1/part-{idx:06}.zip"
                    else:
                        url = f"{baseurl}diffusiondb-large-part-2/part-{idx:06}.zip"
                else:
                    url = f"{baseurl}images/part-{idx:06}.zip"

                loop_file_path = f"{output}part-{idx:06}.zip"
                # It's trying to download the file, and if it encounters an
                # HTTPError, it prints the error.
                try:
                    urlretrieve(url, loop_file_path)
                except HTTPError as e:
                    print(f"HTTPError downloading file: {url} - {e}")
                files_to_unzip.append(loop_file_path)
                # It's writing the url of the file to a manifest file.
                with open("manifest.txt", "a") as f:
                    f.write(url + "\n")
                time.sleep(0.1)
                bar()

    # It's checking if the user wants to unzip the files, and if they do, it
    # returns a list of files to unzip. It would be a bad idea to put these
    # together as the process is already lengthy.
    if unzip and len(files_to_unzip) > 0:
        return files_to_unzip


def unzip_file(file: str):
    """
    > This function takes a zip file as an argument and unpacks it

    :param file: str
    :type file: str
    :return: The file name without the .zip extension
    """
    shutil.unpack_archive(file, extract_dir=file.replace(".zip", ""))
    #delete archive after unpacking
    os.remove(file)
    return f"File: {file.replace('.zip', '')} has been unzipped"


def unzip_all(files: list):
    """
    > Unzip all files in a list of files

    :param files: list
    :type files: list
    """
    with alive_bar(len(files), title="Unzipping files") as bar:
        for file in files:
            unzip_file(file)
            time.sleep(0.1)
            bar()


def main(index=None, range_max=None, output=None, unzip=None, large=None):
    """
    `main` is a function that takes in an index, a range_max, an output, and an
    unzip, and if the user confirms that they have enough space, it downloads
    the files from the index to the output, and if unzip is true, it unzips them

    :param index: The index of the file you want to download
    :param range_max: The number of files to download
    :param output: The directory to download the files to
    :param unzip: If you want to unzip the files after downloading them, set
        this to True
    :param large: If you want to download from DiffusionDB Large (14 million
        images) instead of DiffusionDB 2M (2 million images)
    :return: A list of files that have been downloaded
    """
    if index and range_max:
        if range_max - index >= 1999:
            confirmation = input("Do you have at least 1.7Tb free: (y/n)")
            if confirmation != "y":
                return
        files = download(index, range_max, output, large)
        if unzip:
            unzip_all(files)
    elif index:
        download(index, output=output, large=large)
    else:
        print("No index provided")


# This is a common pattern in Python. It allows you to run the main function of
# your script by running the script through the interpreter. It also allows you
# to import the script into the interpreter without automatically running the
# main function.
if __name__ == "__main__":
    main(index, range_max, output, unzip, large)