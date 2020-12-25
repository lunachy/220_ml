import cv2 as cv
import numpy as np


def template_demo():
    tpl = cv.imread("./shuiyin.jpg")
    # tpl = cv.cvtColor(tpl, cv.COLOR_BGR2GRAY)
    target = cv.imread("./test5.jpg")
    hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    low_hsv = np.array([100,100,160])
    high_hsv = np.array([110,115,170])
    mask = cv.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    cv.imshow("test",mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return
    # target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    # cv.imshow("template image", tpl)
    # cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  # 各种匹配算法
    methods = [cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]  # 获取模板图像的高宽
    for md in methods:
        result = cv.matchTemplate(target, tpl, md)
        # print(result)
        # result是我们各种算法下匹配后的图像
        # cv.imshow("%s"%md,result)
        # 获取的是每种公式中计算出来的值，每个像素点都对应一个值
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        print(min_val, max_val)
        threshold = 0.3
        if max_val > threshold:
            if md == cv.TM_SQDIFF_NORMED:
                tl = min_loc  # tl是左上角点
            else:
                tl = max_loc
            br = (tl[0] + tw, tl[1] + th)  # 右下点
            img = target.copy()
            img = cv.rectangle(img, tl, br, (0, 0, 255), 2)  # 画矩形
            cv.imshow("match-%s" % md, img)
            # cv.imwrite(f"test1__{md}.jpg", img)


# src = cv.imread("./test2.jpg")  # 读取图片
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)  # 创建GUI窗口,形式为自适应
# cv.imshow("input image", src)  # 通过名字将图像和窗口联系
template_demo()
cv.waitKey(0)  # 等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv.destroyAllWindows()  # 销毁所有窗口
