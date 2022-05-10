from pathlib import Path
import cv2
import numpy as np
import random

# 载入图片
# fire = cv2.imread("C:/Users/User/Desktop/JIG/Project/paste_objects/source_images/fire.jpeg")
def read_label(label_path: Path):
    if label_path.exists():
        with open(label_path, "r") as f:
            label = []
            for line in f.readlines():
                label.append(np.array([float(s) for s in line.split(" ")]))
        if len(label):
            return np.stack(label)
    return np.zeros((0, 5))


target_label = read_label(
    Path(
        "C:/Users/User/Desktop/JIG/Project/paste_objects/target_images/NVR_ch4_main_20220304000310_20220304000350_0003.txt"
    )
)
fire_label = read_label(
    Path("C:/Users/User/Desktop/JIG/Project/paste_objects/fire.txt")
)


fire_pic = cv2.imread(
    "C:/Users/User/Desktop/JIG/Project/paste_objects/source_images/fire.jpeg"
)
target_pic = cv2.imread(
    "C:/Users/User/Desktop/JIG/Project/paste_objects/target_images/NVR_ch4_main_20220304000310_20220304000350_0003.png"
)

# 载入后先显示
# cv2.imshow("fire", fire)
# cv2.imshow("target", target)
def cxcywh_to_xyxy(label: np.ndarray) -> np.ndarray:
    category = label[:, 0]  # ndim = 1
    cx = label[:, 1]
    cy = label[:, 2]
    w = label[:, 3]
    h = label[:, 4]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = x1 + w
    y2 = y1 + h
    return np.stack((category, x1, y1, x2, y2), 1)  # ndim = 2 (N, 5)


fire_boxes = cxcywh_to_xyxy(np.array(fire_label))
human_boxes = cxcywh_to_xyxy(np.array(target_label))

# ih = 720
# iw = 1280
th, tw, _ = target_pic.shape
fh, fw, _ = fire_pic.shape
for box in human_boxes:  # boxes (n,5)
    # box [5]
    cat, x1, y1, x2, y2 = box  # print(box)
    human = target_pic[int(y1 * th) : int(y2 * th), int(x1 * tw) : int(x2 * tw)]

# cv2.imshow("human", human)
cv2.imshow("target", target_pic)
"""==============================================================HELP~====↓↓↓↓================================================================"""

fires = {}
for box in range(len(fire_boxes)):  # (3,5)
    cat, x1, y1, x2, y2 = fire_boxes[box]  ## get Unit size of that typical box of fire
    fire_type = fire_pic[
        int(y1 * fh) : int(y2 * fh), int(x1 * fw) : int(x2 * fw)
    ]  ## a type of fires
    print(fire_type.shape)
    fires[f"fire{box+1}"] = fire_type
    cv2.imshow(f"fire{box+1}", fires[f"fire{box+1}"])
"""
    # np.concatenate(fire_type, 0)
    # np.stack((fire_type, x1, y1, x2, y2), 1)
    fboxh, fboxw, _ = fire_type.shape  ## get Actual size of that typical box of fire
    fires = np.zeros((fboxh, fboxw, 3)) + fire_type  ## append all type into 'fires'
"""

# 设置显示区域和位置

# 将原图的蓝色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，将得到的混合结果存到ImgB中
for j in range(3):
    for i in range(len(fires)):
        fh, fw, _ = fires[f"fire{i+1}"].shape
        print(fh, fw)
        rfh = np.random.randint(1, min(721, fh), size=1)
        print(rfh)
        rfw = np.random.randint(1, min(1281, fw), size=1)
        print(rfw)
        rfy = np.random.randint(0, 721 - rfh, size=1)
        print(rfy)
        rfx = np.random.randint(0, 1281 - rfw, size=1)
        print(rfx)
        x2, x1, y2, y1 = rfx + rfw, rfx, rfy + rfh, rfy

        resized_fires = cv2.resize(
            fires[f"fire{i+1}"],
            (x2[0] - x1[0], y2[0] - y1[0]),
            interpolation=cv2.INTER_AREA,
        )
        fires_mask = resized_fires.mean(-1) >= 30  # (box_h, box_w, 1)
        target_pic[y1[0] : y2[0], x1[0] : x2[0]][fires_mask] = resized_fires[fires_mask]
        # cv2.imshow(f"blended{j+1}.{i+1}", target_pic)

cv2.imshow("blended", target_pic)


""" target[:50,:50,][fire_mask, :,:] = fire[fire_mask, :,:]
roi = target[350 : 350 + th, 800 : 800 + tw]
imageROI = cv2.cvtColor(fire, cv2.COLOR_BGR2GRAY)
imageROI = cv2.bitwise_and(roi, roi, mask=imageROI)

# 组合图像
dst = cv2.addWeighted(imageROI,0.7,fire,0.3,0,imageROI)
target[350:350+rows,800:800+cols] = dst
"""
"""=============================="""

# 显示组合结果
# cv2.imshow("blended", target_pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
