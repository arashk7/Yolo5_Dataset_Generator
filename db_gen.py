import urllib.request
import argparse
import pandas as pd
import os
import numpy as np
import cv2 as cv
import imageio

Image_Dim = (416,416)
Output_Dir = 'E:\\Dataset/zhitang'
Dataset_Name = 'Dataset_Zhitang_Yolo5'
# s_img = cv2.imread("smaller_image.png")
# l_img = cv2.imread("larger_image.jpg")
# def image_to_origin_size(img_orig,img_mask,x_offset,y_offset):
#     l_img[y_offset:y_offset+img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
def img_extend(orig, mask_img, x, y):
    '''
    Make a new Mask image with same size of the original image
    Put the mask image into x, y location of the new mask image
    :param orig:
    Original fundus image downloaded directly from DeepDr dataset
    :param mask_img:
    provided disease Mask in the DeepDr dataset
    :param x:
    X location of the mask in the original image
    :param y:
    Y location of the mask in the original image
    :return:
    It returns new Mask image with same size as the original fundus image
    '''
    print(orig.shape)
    print(mask_img.shape)
    new_img = np.zeros(shape=[orig.shape[0], orig.shape[1]], dtype=np.uint8)
    xs = y
    ys = x
    if xs + mask_img.shape[0] > orig.shape[0]:
        xe = orig.shape[0]
    else:
        xe = xs + mask_img.shape[0]

    if ys + mask_img.shape[1] > orig.shape[1]:
        ye = orig.shape[1]
    else:
        ye = ys + mask_img.shape[1]

    new_img[xs:xe, ys:ye] = mask_img[0:xe - xs, 0:ye - ys]
    return new_img
    # cv.imwrite('res.jpg',new_img)


def image_roi(img, x1, y1, x2, y2):
    img = img[x1:x2, y1:y2]
    return img


out_img_dir = 'images'
''' Set directory for downloaded fundus images '''

out_img_resized = 'img_resized'
''' Set directory for bleeding mask after processing '''

out_bld_dir = 'bleedings'
''' Set directory for downloaded bleeding mask images '''

out_bld_dir_extend = 'bld_extend'
''' Set directory for bleeding mask after processing '''

out_bld_dir_resized = 'bld_resized'
''' Set directory for bleeding mask after resizing '''

# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--in_csv_path', type=str, default='./ZhitangSeg1K.csv')
''' Input CSV file'''

parser.add_argument('--out_db_dir', type=str, default='E:\Dataset/zhitang/dataset2/')
''' Output dataset directory '''

arg = parser.parse_args()
''' Process all the input arguments '''

in_csv_path = arg.in_csv_path
if not os.path.exists(in_csv_path):
    raise EOFError('Input CSV file (' + str(in_csv_path) + ') not found')
''' :raise error if the directory does not exist '''
out_db_dir = arg.out_db_dir
if not os.path.exists(out_db_dir):
    raise EOFError('Output directory (' + str(out_db_dir) + ') not found')
''' :raise error if the directory does not exist '''

main_db = pd.read_csv(in_csv_path, keep_default_na=False)
''' Read the CSV file '''
image_urls = main_db['image_url']

print(len(main_db['image_url']))
co = 0
''' record counter '''
for i in range(len(main_db['image_url'])):
    url = main_db['image_url'][i]
    ''' Read the image URL '''
    # bleeding_url = main_db['bl_ink_url'][i]
    ''' Read the bleeding mask URL '''
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    filename = url.split(sep='/')[-1]
    print(filename)
    with open(os.path.join(out_db_dir, out_img_dir, filename), "wb") as f:
        with urllib.request.urlopen(req) as r:
            f.write(r.read())
    orig = cv.imread(os.path.join(out_db_dir, out_img_dir, filename))
    
    resized = cv.resize(orig, Image_Dim, interpolation = cv.INTER_AREA)
    
    status = cv.imwrite(os.path.join(Output_Dir,Dataset_Name,filename),resized)
    
    # if bleeding_url not in (None, ""):
    #     ''' If the mask URL exist '''
    #     bl_x = main_db['bl_point_X'][i]
    #     ''' Read mask location X '''
    #     bl_y = main_db['bl_point_Y'][i]
    #     ''' Read mask location Y '''
    #     if int(bl_x) == -1 or int(bl_y) == -1:
    #         continue
        
    #     req = urllib.request.Request(bleeding_url, headers={'User-Agent': 'Mozilla/5.0'})
    #     filename_s = bleeding_url.split(sep='/')[-1]
    #     with open(os.path.join(out_db_dir, out_bld_dir, filename_s), "wb") as f:
    #         with urllib.request.urlopen(req) as r:
    #             f.write(r.read())

    #     limg = cv.imread(os.path.join(out_db_dir, out_img_dir, filename))
    #     ''' Read the downloaded fundus image using OpenCV API '''
    #     simg = imageio.mimread(os.path.join(out_db_dir, out_bld_dir, filename_s))
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
