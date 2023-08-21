## Numpy基础

### ndarray：一种多维数组对象

NumPy最重要的一个特点就是其N维数组对象（即ndarray），该对象是一个快速而灵活的大数据集容器。你可以利用这种数组对整块数据执行一些数学运算，其语法跟标量元素之间的运算一样。

1. 创建列表`np.arange(0,10)`: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

   ​	  `np.arange(0,10,2)`: array([0, 2, 4, 6, 8])

2. 创建矩阵`np.ones()`, `np.zeros`

3. 随机列表`np.random.randint(0,100,10)`: array([62, 95, 82, 46, 59, 72, 18, 49, 85, 42])

4. ndarray常用函数: 

   | 名称     | 函数                   | 示例                                    |
   | -------- | ---------------------- | --------------------------------------- |
   | 最值     | `max()`, `min()`       | arr.max()                               |
   | 最值索引 | `argmax()`, `argmin()` | arr.argmax()                            |
   | 平均值   | `mean()`               | arr.mean()                              |
   | 形状     | `shape`                | arr.shape                               |
   | 变形     | `reshape()`            | matrix = np.arange(0,25).reshape((5,5)) |

## 操作图像

### 使用 matplotlib 

```py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```

1. 从文件打开：`img = Image.open('bus.jpg')`

2. 转换数据类型：`img_arr = np.asarray(img)`

3. 显示图像：`plt.imshow(img_arr)`

### 使用 OpenCV

```py
import numpy as np
import matplotlib.pyplot as plt
import cv2
```

1. #### 从文件打开：

   ```py
   img = cv.imread('zidane.jpg')
   img_fixed = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   plt.imshow(img_fixed)
   ```

   - [x] `cv.COLOR_BGR2RGB`:因为 OpenCV 的图像通道显示顺序为 ==BGR==，matplotlib 通道显示顺序是==RGB==，要使用`plt.imshow()`显示出正常的图像，需要转换通道顺序才可以。

2. #### 缩放图片：

   - [x] `cv2.resize(img_fixed, (500,100))`：改变图片长为 500，高为 100。

3. #### 翻转图片：

   - [x] `img_fixed, 1)`: ==0==垂直翻转, ==1==水平翻转, ==-1==水平垂直都翻转。

4. #### 保存图片：

   ```py
   img_save = cv2.cvtColor(img_flip, cv2.COLOR_RGB2BGR)
   cv2.imwrite('./flip_img.jpg', img_save)
   ```

   - [x] `cv2.COLOR_RGB2BGR`: 使用 OpenCV 操作后的图片若需要保存，则需要再经过一次通道顺序转换。

## 绘制图形

1. **矩形:** cv2.rectangle(*img*=img_black, *pt1*=(100,100), *pt2*=(400,300), *color*=(0,255,0), *thickness*=10)

   > 要绘制矩形，你需要设置矩形的左上角和右下角的坐标。这次我们将在图像的右上角绘制一个绿色矩形。 参考一下代码：

2. **圆形**：cv2.circle(*img*=img_black,*center*=(500,200),*radius*=50,*color*=(0,255,255),*thickness*=-1)

3. **直线**：cv2.line(img_black,(100,400),(600,600),(255,0,0),10)        

4. **文字**：cv2.putText(img_black, "text", (200,600), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255, 255, 255),5, cv2.LINE_AA)

   > 要将文本放入图像中，你需要传递以下几个参数： - 要写入的文本数据， - 要放置的位置（即文本数据的左下角） - 字体类型（检查cv.putText（）文档以获取支持的字体）， - 字体大小 - 常规的参数，比如颜色、粗细、线型等，为了更好看，建议使用`lineType = cv.LINE_AA`作为线型参数的值。

## 操作视频流

### 读取视频

```py
import cv2

cap = cv2.VideoCapture('output.avi')
if not cap.isOpened():
    print("The file does not exist or the encoding is wrong.")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```

`cap.read()`返回一个bool值（True / False）。如果读取帧正确，则它将为True。因此，你可以通过值来确定视频的结尾。 如果初始化摄像头失败，上面的代码会报错。你可以使用`cap.isOpened()`来检查是否初始化。如果返回值是True，说明初始化成功，否则就要使用函数 `cap.open()`。

你还可以使用`cap.get(propld)`方法访问此视频的某些功能，参数propId代表0到18之间的数字。每个数字表示视频的一个属性，详细的信息参见：[cv::VideoCapture::get()](https://docs.opencv.org/4.0.0/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939).其中一些值可以使用`cap.set(propId，value)`进行修改。其中参数value是你想要的新值。

例如，我可以通过`cap.get(cv.CAP_PROP_FRAME_WIDTH)`和`cap.get(cv.CAP_PROP_FRAME_HEIGHT)`分别检查帧宽和高度。它返回给我默认值640x480。但如果我想将其修改为320x240，只需使用`ret=cap.set(cv.CAP_PROP_FRAME_WIDTH，32)`和`ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT，240)` 。

> 与从相机捕获视频原理相同，只需将设备索引更改为视频文件的名字。同时在显示帧时，请给cv.waitKey()函数传递适当的时间参数。如果它太小，视频将非常快，如果它太高，视频将会很慢。在正常情况下，25毫秒就可以了。

### 保存视频

```py
import cv2

cap = cv2.VideoCapture(1)

# 保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID,MJPG

fps = 20
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
print("fps:", fps, "\nwidth:", w, "\nheight:", h)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

```

首先创建一个==VideoWriter==对象，我们应该指定输出文件名（例如：`output.avi`），然后我们应该指定FourCC代码并传递每秒帧数（fps）和帧大小。最后一个是isColor标志，如果是True，则每一帧是彩色图像，否则每一帧是灰度图像。 FourCC是用于指定视频编解码器的4字节代码： 

- 在Fedora中：DIVX，XVID，MJPG，X264，WMV1，WMV2。（XVID更为可取.MJPG会产生高大小的视频.X264提供非常小的视频） 
- 在Windows中：DIVX（更多要测试和添加） 
- 在OSX中：MJPG（.mp4），DIVX（.avi），X264（.mkv）。

从相机捕获图像之后，在垂直方向上翻转每一帧之后逐帧保存。

### 视频上画图

> 掌握函数：`cv.line()`, `cv.circle()` , `cv.rectangle()`, `cv.ellipse()`, `cv.putText()`

```py
import time
import cv2

cap = cv2.VideoCapture(1)
start_time = time.time()

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        # 计算帧率
        now = time.time()
        fps = int(1 / (now - start_time))
        start_time = now

        fps_text = "fps:" + str(fps)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps_text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()		
```



