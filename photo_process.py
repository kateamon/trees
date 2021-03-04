"""
This Python script takes photos from a source directory:
    Flips those image left-right
    Saves the flipped images into a separate output directory,
    with _flip appended to the original filename before .jpg.

Use glob2 third-party library to generate a list of all jpg image filenames.
The string indexing [:-4] gets the filename cutting off the .jpg extension minus last 4 characters.

Image processing can be also be used to resize images to 64x64 pixel resolution. But we don't need to resize, just keeping for reference.

"""

import glob2
# import cv2  # OpenCV is a pain to install
# Using the Python Imaging Library (PIL) instead of OpenCV
import PIL
from PIL import Image
import numpy
# for log file
from datetime import datetime as dt

# Read in the tree image jpg files
# Get the Pepper Tree images
SOURCE_DIR_PEPPER = "/home/kate/data/trees-raw-data/california-pepper-tree/"
image_files_pepper = glob2.glob("/home/kate/data/trees-raw-data/california-pepper-tree/*.jpg")

# Get the Weeping Willow tree image files
SOURCE_DIR_WILLOW = "/home/kate/data/trees-raw-data/Lovely-trees-weeping-willow/"
image_files_willow = glob2.glob("/home/kate/data/trees-raw-data/Lovely-trees-weeping-willow/*.jpg")

# Where to put the flipped images:
OUTPUT_DIR_PEPPER = "/home/kate/data/trees-processed/pepper-trees/"
OUTPUT_DIR_WILLOW = "/home/kate/data/trees-processed/weeping-willow-trees/"


def flip_all_images(image_files, SOURCE_DIR, OUTPUT_DIR):
    number_of_photos = len(image_files)
    for filename in image_files:
        img = Image.open(filename)
        img_flip = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Determine how many characters in the source directory
        len_source = len(SOURCE_DIR) 

        # Put the input directory path in, strip out the source directory path from the new filename
        newfilename = OUTPUT_DIR + filename[len_source:-4] + "_flip" + ".jpg"
        # print(newfilename)
        img_flip.save(newfilename)
    #create a log file to also put in the output directory

    log_date = dt(dt.now().year, dt.now().month, dt.now().day, dt.now().hour, dt.now().minute)

    line0 = "Created by function flip_all_images. Date/Time of log file: " + str(log_date)
    line1 = "Done flipping and saving " + str(number_of_photos) +" photos."
    line2 = "From photos in input directory " + SOURCE_DIR
    line3 = "Flipping and saving those photos to output directory: " + OUTPUT_DIR

    log_file_name = OUTPUT_DIR + "000_flip_all_images.log"
    log_file_name = open(log_file_name, "w")
    log_file_name.write(line0 + "\n")
    log_file_name.write(line1 + "\n")
    log_file_name.write(line2 + "\n")
    log_file_name.write(line3 + "\n")
    log_file_name.close()

    print(line1)
    print(line2)
    print(line3)
    print("")

# Now run the flipping function on both tree photo sources
flip_all_images(image_files_pepper, SOURCE_DIR_PEPPER, OUTPUT_DIR_PEPPER)
flip_all_images(image_files_willow, SOURCE_DIR_WILLOW, OUTPUT_DIR_WILLOW)

# ~~~~~~~~~~~~~~~~~~~ old stuff for reference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# image_files = glob2.glob("*.jpg")

# for filename in image_files:
#     img = cv2.imread(filename, 1)
#     img64x64 = cv2.resize(img, (64, 64))
#     newfilename = filename[:-4] + "64x64" + ".jpg"
#     cv2.imshow(newfilename, img100x100)
#     cv2.waitKey(2000)
#     cv2.imwrite(newfilename, img100x100)

# import PIL
# from PIL import Image
#
# #read the image
# im = Image.open("sample-image.png")
#
# #flip image
# out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
# out.save('transpose-output.png')
