import cv2
import numpy as np
import os
from skimage import exposure
import skimage.morphology as morphology

def noise_remove_cv2(gray_img, k):
    """
    8邻域降噪
    Args:
        image_name: 图片文件命名
        k: 判断阈值

    Returns:

    """

    def calculate_noise_count(img_obj, w, h):
        """
        计算邻域非白色的个数
        Args:
            img_obj: img obj
            w: width
            h: height
        Returns:
            count (int)
        """
        count = 0
        width, height = img_obj.shape
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
                    count += 1
        return count

    # 灰度
    w, h = gray_img.shape
    for _w in range(w):
        for _h in range(h):
            if _w == 0 or _h == 0:
                gray_img[_w, _h] = 255
                continue
            # 计算邻域pixel值小于255的个数
            pixel = gray_img[_w, _h]
            if pixel == 255:
                continue

            if calculate_noise_count(gray_img, _w, _h) < k:
                gray_img[_w, _h] = 255

    return gray_img


def grey_scale(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (img_gray - A) +0.9)
    return output

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


#局部阈值
def local_threshold(gray):
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)

if __name__ == '__main__':
    image_dir = r"D:\workspace\Train\captcha2"
    right_num = 0
    for i, p in enumerate(os.listdir(image_dir)):
        n = os.path.join(image_dir, p)
        print(n)
        src = cv2.imread(n,cv2.IMREAD_GRAYSCALE)
        cv2.imshow('src', src)

        gamma_img = exposure.adjust_gamma(src, 5)
        image = noise_remove_cv2(gamma_img, 2)
        cv2.imshow('gamma_img', gamma_img)
        cv2.imshow('remove_image', image)
        img2 = morphology.remove_small_objects(src, 15,2)
        cv2.imshow('remove', img2)
        cv2.waitKey(1000)


