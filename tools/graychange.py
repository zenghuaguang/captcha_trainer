'''灰度变换'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''反转变换'''
def reverse(img):
    output = 255 - img
    return output
'''对数变换'''

def log(c, img):
    output_img = c*np.log(1.0+img)
    output_img = np.uint8(output_img+0.5)
    return output_img

"""幂律变换（伽马）"""
def gamma(img, c, v=1):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img+0.5)  # 这句一定要加上
    return output_img

def gamma_plot(c, v):
    x = np.arange(0, 256, 0.01)
    y = c*x**v
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('伽马变换函数')
    plt.xlim([0, 255]), plt.ylim([0, 255])
    plt.show()

"""分段线性变换Segmental Linear Transformation"""
def SLT(img, x1, x2, y1, y2):
    lut = np.zeros(256)
    for i in range(256):
            if i < x1:
                lut[i] = (y1/x1)*i
            elif i < x2:
                lut[i] = ((y2-y1)/(x2-x1))*(i-x1)+y1
            else:
                lut[i] = ((y2-255.0)/(x2-255.0))*(i-255.0)+255.0
    img_output = cv2.LUT(img, lut)
    img_output = np.uint8(img_output+0.5)
    return img_output

"""灰度级分层"""
def GrayLayer(img):
    lut = np.zeros(256, dtype=np.uint8)
    layer1 = 30
    layer2 = 60
    value1 = 10
    value2 = 250
    for i in range(256):
        if i >= layer2:
            lut[i] = value1
        elif i >= layer1:
            lut[i] = value2
        else:
            lut[i] = value1
    ans = cv2.LUT(img, lut)
    return ans

'''灰度拉伸 '''
def GrayLayer(img):
    lut = np.zeros(256, dtype=np.uint8)
    layer1 = 30
    layer2 = 60
    value1 = 10
    value2 = 250
    for i in range(256):
        if i >= layer2:
            lut[i] = value1
        elif i >= layer1:
            lut[i] = value2
        else:
            lut[i] = value1
    ans = cv2.LUT(img, lut)
    return ans
"""最大最小值拉伸"""
def max_min_strech(img):
    max1 = np.max(img)
    min1 = np.min(img)
    output_img = (255.0*(img-min1))/(max1-min1)  # 注意255.0 而不是255 二者算出的结果区别很大
    output_img1 = np.uint8(output_img+0.5)
    return output_img1


if __name__ == '__main__':
    img_input = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('input', img_input)
    img_output = gamma(img_input,1.5)
    cv2.imshow('output', img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()