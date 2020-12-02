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

Classes = {'bleeding':0,'2':1}
# s_img = cv2.imread("smaller_image.png")
# l_img = cv2.imread("larger_image.jpg")
# def image_to_origin_size(img_orig,img_mask,x_offset,y_offset):
#     l_img[y_offset:y_offset+img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
# def img_extend(orig, mask_img, x, y):
#     '''
#     Make a new Mask image with same size of the original image
#     Put the mask image into x, y location of the new mask image
#     :param orig:
#     Original fundus image downloaded directly from DeepDr dataset
#     :param mask_img:
#     provided disease Mask in the DeepDr dataset
#     :param x:
#     X location of the mask in the original image
#     :param y:
#     Y location of the mask in the original image
#     :return:
#     It returns new Mask image with same size as the original fundus image
#     '''
#     print(orig.shape)
#     print(mask_img.shape)
#     new_img = np.zeros(shape=[orig.shape[0], orig.shape[1]], dtype=np.uint8)
#     xs = y
#     ys = x
#     if xs + mask_img.shape[0] > orig.shape[0]:
#         xe = orig.shape[0]
#     else:
#         xe = xs + mask_img.shape[0]
#
#     if ys + mask_img.shape[1] > orig.shape[1]:
#         ye = orig.shape[1]
#     else:
#         ye = ys + mask_img.shape[1]
#
#     new_img[xs:xe, ys:ye] = mask_img[0:xe - xs, 0:ye - ys]
#     return new_img
#     # cv.imwrite('res.jpg',new_img)
#
#
# def image_roi(img, x1, y1, x2, y2):
#     img = img[x1:x2, y1:y2]
#     return img


# out_img_dir = 'images'
# ''' Set directory for downloaded fundus images '''

# out_img_resized = 'img_resized'
# ''' Set directory for bleeding mask after processing '''

# out_bld_dir = 'bleedings'
# ''' Set directory for downloaded bleeding mask images '''

# out_bld_dir_extend = 'bld_extend'
# ''' Set directory for bleeding mask after processing '''

# out_bld_dir_resized = 'bld_resized'
# ''' Set directory for bleeding mask after resizing '''

# # Read Arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--in_csv_path', type=str, default='./ZhitangSeg1K.csv')
# ''' Input CSV file'''

# parser.add_argument('--out_db_dir', type=str, default='E:\Dataset/zhitang/dataset2/')
# ''' Output dataset directory '''

# arg = parser.parse_args()
# ''' Process all the input arguments '''

# in_csv_path = arg.in_csv_path
# if not os.path.exists(in_csv_path):
#     raise EOFError('Input CSV file (' + str(in_csv_path) + ') not found')
# ''' :raise error if the directory does not exist '''
# out_db_dir = arg.out_db_dir
# if not os.path.exists(out_db_dir):
#     raise EOFError('Output directory (' + str(out_db_dir) + ') not found')
# ''' :raise error if the directory does not exist '''

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

    '''Resizing Image to specified size and calc the aspect ratio'''
    resized = cv.resize(orig, Image_Dim, interpolation=cv.INTER_AREA)
    ar_width = resized.shape[1] / orig.shape[1]
    ar_height = resized.shape[0] / orig.shape[0]

    '''Save the resized image to the specified directory'''
    status = cv.imwrite(os.path.join(Output_Dir, Dataset_Name, 'Images', filename), resized)

    '''Reading different lesion url from the dataframe'''
    ''' Read the bleeding mask URL '''
    bleeding_url = main_db['bl_ink_url'][i]

    ''' Open a text file to records all the labels'''
    arr=filename.split(sep='.')
    filename_new = ''.join(arr[0:-1])+'.txt'
    print(filename_new)
    labels_file = open(os.path.join(Output_Dir,Dataset_Name,'Labels',filename_new), 'w')

    ''' Check the Bleeding mask URL exist '''
    if bleeding_url not in (None, ""):

        ''' Read mask location X '''
        bl_x_orig = main_db['bl_point_X'][i]

        ''' Read mask location Y '''
        bl_y_orig = main_db['bl_point_Y'][i]

        '''Check whether the position is wronge'''
        if int(bl_x_orig) == -1 or int(bl_y_orig) == -1:
            continue

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


        '''Find Contours (find all the bleeding parts in the mask image)'''
        bld_gray = cv.cvtColor(bld_mask, cv.COLOR_RGB2GRAY)
        ret, bld_thresh = cv.threshold(bld_gray, 10, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(bld_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            bl_x_box = bl_x_orig + x
            bl_y_box = bl_y_orig + y

            '''Convert X, Y , Width, Height to the new Scale'''
            bl_x = bl_x_box * ar_width
            bl_y = bl_y_box * ar_height
            bl_w = w * ar_width
            bl_h = h * ar_height

            print(bl_x)
            print(bl_y)

            orig = cv.rectangle(orig, rec=(int(bl_x_box), int(bl_y_box), int(w), int(h)), color=(255, 255, 0), thickness=1)
            resized = cv.rectangle(resized, rec=(int(bl_x), int(bl_y), int(bl_w), int(bl_h)), color=(255, 0, 0),thickness=1)

            ''' Write the bounding box into the file '''
            labels_file.write(str(Classes['bleeding'])+' '+str(bl_x)+' '+str(bl_y)+ ' '+ str(bl_w)+ ' '+str(bl_h)+'\n')


            plt.imshow(orig)
            plt.show()
            plt.imshow(resized)
            plt.show()
            print('okay')
        input()
        labels_file.close()

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
