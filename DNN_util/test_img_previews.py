"""
    This script is for preview the test sets to determin which image has
    whole skeleton
"""
import cv2 as cv
import numpy as np
import config as cfg

if __name__ == '__main__':
    # Read the image txt file to get test lists

    test_list_file = open('../'+cfg.test_list_path, 'r')

    # Some variables
    line_count = 0

    test_files = {}

    # Read line-by-line to parsing attribute files
    while 1:

        # Read into buffer to speed up
        lines = test_list_file.readlines(100000)

        if not lines:
            break
        for line in lines[1:]:

            # Split the line by ','
            tokens = line.split(',')

            image_name = tokens[0]
            attribute = int(tokens[1])

            # Check if the image is negative example
            if attribute is 0:
                continue

            # Now we add to a dictionary
            if not test_files.has_key(image_name):
                test_files[image_name] = 1

    # End of add test_files dictionary
    test_files = test_files.keys()
    intact_flags = np.zeros(len(test_files))
    current_index = 0

    cv.namedWindow('test')

    while current_index < len(test_files):

        # Read image from file
        img = cv.imread(cfg.image_directory + test_files[current_index] + '.jpg', cv.IMREAD_COLOR)

        # Show the image
        cv.imshow('test', img)
        a = cv.waitKey(0)

        if a == 1113939:    # next
            current_index += 1
        elif a == 1113937:  # previous
            current_index -= 1
        elif a == 1048695:  # Whole flag
            intact_flags[current_index] = 2     # 2 means whole
            current_index += 1
        elif a == 1048693:
            intact_flags[current_index] = 1     # 1 means upper
            current_index += 1
        elif a == 1048686:  # Not flag
            intact_flags[current_index] = 0
            current_index += 1
        elif a == 1048689:
            break

    cv.destroyWindow('test')

    # write to file
    flag_file = open('../test_flag.txt', 'w')

    for i in range(0, len(intact_flags), 1):
        img_name = test_files[i]
        flag = intact_flags[i]
        flag_file.write('%s: %d\n' % (img_name, flag))

    flag_file.close()





