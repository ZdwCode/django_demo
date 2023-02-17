import os
import cv2
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):
    ################
    # 获取人的脸部信息，并保存到所属文件夹
    ################
    #检查输入路径是否存在——不存在就创建
    CreateFolder(path_name)

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 1600, 900)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸检测
    classfier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        grey = frame
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                #cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                if w > 200:
                    # 将当前帧保存为图片
                    img_name = '%s\%d.jpg' % (path_name, num)
                    #image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    image = grey[y:y+h,x:x+w]           #保存灰度人脸图
                    cv2.imwrite(img_name, image)
                    num += 1
                    # if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                    #     break

                    #画出矩形框的时候稍微比识别的脸大一圈
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                    # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): break

        # 显示图像
        cv2.imshow(window_name, frame)
        #按键盘‘Q’中断采集
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

def get_face(name):
    ###################################################
    # 相当于公司人力组织一次所有员工人脸信息采集
    ###################################################
    # 员工姓名(要输入英文，汉字容易报错)
    new_user_name = name
    # 采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢
    window_name = 'face_collection'  # 图像窗口
    camera_id = 0 # 相机的ID号
    images_num = 500  # 采集图片数量
    path = './app01/face_recognize/face_image/data/' + new_user_name  # 图像保存位置
    CatchPICFromVideo(window_name, camera_id, images_num, path)

def CreateFolder(path):
    """
    判断地址是否存在
    :param path:
    :return:
    """
    #去除首位空格
    del_path_space = path.strip()
    #去除尾部'\'
    del_path_tail = del_path_space.rstrip('\\')
    #判读输入路径是否已存在
    isexists = os.path.exists(del_path_tail)
    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False

if __name__ == '__main__':
    # if len(sys.argv) != 1:
    #     print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    # else:
    #     CatchUsbVideo("识别人脸区域", 0)
    get_face()