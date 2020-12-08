import urllib.request
import argparse
import pandas as pd
import os
import numpy as np
import cv2 as cv
import imageio
import matplotlib.pyplot as plt

Image_Dim = (416, 416)
Output_Dir = 'E:\\Dataset/zhitang'
Dataset_Name = 'Dataset_Zhitang_Yolo5'

Classes = {'bleeding': 0, 'microaneurism': 1, 'soft_exudate': 2, 'hard_exudate': 3, 'IRMA': 4, 'laser_spot': 5,
           'new_blood_vessels': 6, 'optic_disk': 7, 'macular': 8}

is_display = True

'''Cut a part of the input image'''


def image_roi(img, x1, y1, x2, y2):
    img = img[x1:x2, y1:y2]
    return img


main_db = pd.read_csv('ZhitangSeg1K.csv', keep_default_na=False)
''' Read the CSV file '''
image_urls = main_db['image_url']

print(len(main_db['image_url']))
co = 0
''' record counter '''
for i in range(len(main_db['image_url'])):
    url = main_db['image_url'][i]
    ''' Read the image URL '''

    ''' Downloading Orig Image'''
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    filename = url.split(sep='/')[-1]
    print(filename)
    with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename), "wb") as f:
        with urllib.request.urlopen(req) as r:
            f.write(r.read())
    orig = cv.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename))

    ret, thresh = cv.threshold(orig[:, :, 1], 10, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_cnt)
    x1 = x - 70
    y1 = 0
    x2 = x + w + 70
    y2 = orig.shape[0]
    orig = image_roi(orig, y1, x1, y2, x2)

    '''Resizing Image to specified size and calc the aspect ratio'''
    resized = cv.resize(orig, Image_Dim, interpolation=cv.INTER_AREA)
    ar_width = resized.shape[1] / orig.shape[1]
    ar_height = resized.shape[0] / orig.shape[0]

    '''Save the resized image to the specified directory'''
    status = cv.imwrite(os.path.join(Output_Dir, Dataset_Name, 'Images', filename), resized)

    ''' Open a text file to records all the labels'''
    arr = filename.split(sep='.')
    filename_new = ''.join(arr[0:-1]) + '.txt'
    print(filename_new)
    labels_file = open(os.path.join(Output_Dir, Dataset_Name, 'Labels', filename_new), 'w')

    '''Reading different lesion url from the dataframe'''
    ''' Read the bleeding mask URL '''
    bleeding_url = main_db['bl_ink_url'][i]
    microaneurism_url = main_db['ma_ink_url'][i]
    soft_exudate_url = main_db['se_ink_url'][i]
    hard_exudate_url = main_db['he_ink_url'][i]
    irma_url = main_db['irma_ink_url'][i]
    laser_url = main_db['laser_ink_url'][i]
    nbv_url = main_db['nbv_ink_url'][i]
    optic_disk_url = main_db['od_ink_url'][i]
    macular_url = main_db['mac_ink_url'][i]

    ''' Check the Bleeding mask URL exist '''
    if bleeding_url not in (None, ""):

        ''' Read mask location X '''
        bl_x_orig = main_db['bl_point_X'][i]

        ''' Read mask location Y '''
        bl_y_orig = main_db['bl_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(bl_x_orig) == -1 or int(bl_y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        bl_x_orig -= x1
        bl_y_orig -= y1

        '''Downloading the Bleeding Mask '''
        req = urllib.request.Request(bleeding_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = bleeding_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        bld_mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        '''Get the first frame of the Gif file'''
        # bld_mask = bld_mask[0]
        ''' Get Width and Height of the Mask area'''
        bl_w_orig = bld_mask.shape[1]
        bl_h_orig = bld_mask.shape[0]

        '''Converting images to gray and threshold them to find the masked area'''
        bld_gray = cv.cvtColor(bld_mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        '''Morphological Closing (Dilation + Erosion) to fill the holes in the mask'''
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        '''Find Contours (find all the bleeding parts in the mask image)'''
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = bl_x_orig + x
            y_box = bl_y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            # print(bl_x)
            # print(bl_y)

            # resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 0, 0),thickness=1)

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['bleeding']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')
            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 0, 0),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 0, 0),
                                       thickness=2)

    ''' Check the Microaneurism mask URL exist '''
    if microaneurism_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['ma_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['ma_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the Bleeding Mask '''
        req = urllib.request.Request(microaneurism_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = microaneurism_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['microaneurism']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(0, 255, 0),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(0, 255, 0),
                                       thickness=2)

    ''' Check the soft_exudate mask URL exist '''
    if soft_exudate_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['se_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['se_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(soft_exudate_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = soft_exudate_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['soft_exudate']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(0, 0, 255),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(0, 0, 255),
                                       thickness=2)

    ''' Check the hard_exudate mask URL exist '''
    if hard_exudate_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['he_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['he_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(hard_exudate_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = hard_exudate_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['hard_exudate']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 255, 0),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 255, 0),
                                       thickness=2)

    ''' Check the irma mask URL exist '''
    if irma_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['irma_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['irma_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(irma_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = irma_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['IRMA']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 0, 255),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 0, 255),
                                       thickness=2)
    ''' Check the laser spot mask URL exist '''
    if laser_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['laser_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['laser_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(laser_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = laser_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['laser_spot']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(0, 255, 255),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(0, 255, 255),
                                       thickness=2)
    ''' Check the new blood vessel mask URL exist '''
    if nbv_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['nbv_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['nbv_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(nbv_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = nbv_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_box = x_orig + x
            y_box = y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = x_box * ar_width
            bl_y = y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            '''Normalize the results'''
            bl_x_n = bl_x / Image_Dim[0]
            bl_y_n = bl_y / Image_Dim[1]
            bl_w_n = bl_w / Image_Dim[0]
            bl_h_n = bl_h / Image_Dim[1]

            ''' Write the bounding box into the file '''
            labels_file.write(
                str(Classes['new_blood_vessels']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(
                    bl_w_n) + ' ' + str(
                    bl_h_n) + '\n')

            if is_display:
                orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 255, 255),
                                    thickness=2)
                resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 255, 255),
                                       thickness=2)
    ''' Check the optic dist mask URL exist '''
    if optic_disk_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['od_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['od_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue

        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(optic_disk_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = optic_disk_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        max_cnt = max(contours, key=cv.contourArea)
        # for cnt in contours:
        x, y, w, h = cv.boundingRect(max_cnt)
        x_box = x_orig + x
        y_box = y_orig + y

        '''Convert X, Y , Width, Height to the new Scale'''
        bl_x = x_box * ar_width
        bl_y = y_box * ar_height
        bl_w = w * ar_width
        bl_h = h * ar_height

        '''Normalize the results'''
        bl_x_n = bl_x / Image_Dim[0]
        bl_y_n = bl_y / Image_Dim[1]
        bl_w_n = bl_w / Image_Dim[0]
        bl_h_n = bl_h / Image_Dim[1]

        ''' Write the bounding box into the file '''
        labels_file.write(
            str(Classes['optic_disk']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                bl_h_n) + '\n')

        if is_display:
            orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 100, 255),
                                thickness=2)
            resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 100, 255),
                                thickness=2)

    ''' Check the macular mask URL exist '''
    if macular_url not in (None, ""):

        ''' Read mask location X '''
        x_orig = main_db['mac_point_X'][i]

        ''' Read mask location Y '''
        y_orig = main_db['mac_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(x_orig) == -1 or int(y_orig) == -1:
            continue
        '''convert the pose to new ROI image'''
        x_orig -= x1
        y_orig -= y1

        '''Downloading the  Mask '''
        req = urllib.request.Request(macular_url, headers={'User-Agent': 'Mozilla/5.0'})
        filename_s = macular_url.split(sep='/')[-1]
        with open(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s), "wb") as f:
            with urllib.request.urlopen(req) as r:
                f.write(r.read())

        '''Open the downloaded Mask file'''
        mask = imageio.imread(os.path.join(Output_Dir, Dataset_Name, 'temp', filename_s))

        ''' Get Width and Height of the Mask area'''
        w_orig = mask.shape[1]
        h_orig = mask.shape[0]

        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bld_thresh = cv.morphologyEx(bld_thresh, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv.contourArea)
        # for cnt in contours:
        x, y, w, h = cv.boundingRect(max_cnt)
        x_box = x_orig + x
        y_box = y_orig + y

        '''Convert X, Y , Width, Height to the new Scale'''
        bl_x = x_box * ar_width
        bl_y = y_box * ar_height
        bl_w = w * ar_width
        bl_h = h * ar_height

        '''Normalize the results'''
        bl_x_n = bl_x / Image_Dim[0]
        bl_y_n = bl_y / Image_Dim[1]
        bl_w_n = bl_w / Image_Dim[0]
        bl_h_n = bl_h / Image_Dim[1]

        ''' Write the bounding box into the file '''
        labels_file.write(
            str(Classes['macular']) + ' ' + str(bl_x_n) + ' ' + str(bl_y_n) + ' ' + str(bl_w_n) + ' ' + str(
                bl_h_n) + '\n')

        if is_display:
            orig = cv.rectangle(orig, rec=(int(x_box), int(y_box), int(w), int(h)), color=(255, 255, 100),
                                thickness=2)
            resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 255, 100),
                                   thickness=2)

    orig = cv.cvtColor(orig, cv.COLOR_RGB2BGR)
    resized = cv.cvtColor(resized, cv.COLOR_RGB2BGR)
    # plt.imshow(orig)
    # plt.show()
    plt.imshow(resized)
    plt.show()
    labels_file.close()
    # input()
    #     limg = cv.imread(os.path.join(out_db_dir, out_img_dir, filename))
    #     ''' Read the downloaded fundus image using OpenCV API '''
    #     
    #     ''' Read the downloaded mask image using OpenCV API '''
    #     simg = simg[0]
    #     ''' Get the first frame of the GIF file '''
    #     simg = cv.cvtColor(simg, cv.COLOR_RGB2GRAY)
    #     ''' Convert the mask image to gray '''
    #     ret, simg = cv.threshold(simg, 10, 255, cv.THRESH_BINARY)
    #     ''' Make the mask image to binary (Black and White) image'''

    #     bld_extend = img_extend(orig=limg, mask_img=simg, x=int(bl_x), y=int(bl_y))
    #     ''' Convert the mask image to mask resized frame (resized to the fundus image size) '''
    #     cv.imwrite(os.path.join(out_db_dir, out_bld_dir_extend, filename), bld_extend)
    #     ''' Record the result in the output directory '''
    #     co += 1
    #     ''' Count the masks'''
    #     print(limg.shape)
    #     ret,thresh = cv.threshold(limg[:,:,1],10,255,cv.THRESH_BINARY)
    #     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     areas = [cv.contourArea(c) for c in contours]
    #     max_index = np.argmax(areas)
    #     cnt = contours[max_index]
    #     x, y, w, h = cv.boundingRect(cnt)

    #     limg = image_roi(limg,0,x-50,limg.shape[0],x+w+50)
    #     cv.imwrite(os.path.join(out_db_dir, out_img_resized, filename), limg)

    #     bld_resized = image_roi(bld_extend, 0, x - 50, bld_extend.shape[0], x + w + 50)
    #     cv.imwrite(os.path.join(out_db_dir, out_bld_dir_resized, filename), bld_resized)

print(co)
