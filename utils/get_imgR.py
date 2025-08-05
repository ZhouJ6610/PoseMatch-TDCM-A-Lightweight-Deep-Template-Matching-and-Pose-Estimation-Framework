# 生成带有旋转角度的目标框算法：

# 步骤一. 随机角度旋转带有目标框(bbox)图像 --- 得到imgR, bboxR
# 步骤二. 由于旋转后四个角落会产生空白，需要进行裁剪
   # 裁剪要求：不能含有空白部分，包含完整的边界框，剪切后图像大于size：224*224
   # 裁剪算法：
	# 计算目标框bboxR的最小外接矩形，
	# 1）外接矩形已经有部分超出imgR，即与空白部分有相交，回到步骤一（最多10次尝试）
	# 2）仍然在imgR中，则开始扩大剪切区域，计算每条边S可以扩张的距离maxD_S(受size和目标框bboxR限制)
	   # 即：移动后S与imgR边界的两个交点距离需等于224 (可移动距离为含有0，回到步骤一)
	   # a. 随机选择一条边s向外扩张（移动），移动距离为计算出的最大范围内随机【0，maxD_s】, 同时确认出对边的位置
	   # b. 移动后需要更新其邻边的最大移动距离
		# -- 相邻的两条边的最大扩张距离受到S与imgR产生的两个新交点
		# -- 更新可移动距离含有0，回到步骤a
	   # c. 在剩余的邻边里随机选择一边重复步骤ab
	   # d. 剪切扩张后的矩形

import numpy as np
import cv2, copy, math
import matplotlib.pyplot as plt


def rotate_crop(image, bbox, max_attempts: int = 10):
    for i in range(max_attempts):
        # 步骤一：随机旋转图像和边界框
        angle = np.random.uniform(-180, 180)  # 随机旋转角度，可调整范围
        img_R, bbox_R = rotate(image, bbox, angle)

        # 计算旋转后边界框的最小外接矩形
        min_rect = get_min_rect(bbox_R)
        rect = min_rect.copy()
        
        # 检查边界框是否与空白区域相交
        if rect_in_imgR(rect, img_R):
            # 步骤二：扩张裁剪区域
            crop_rect = expand_rect(rect, img_R, angle, size=224)
            if crop_rect is not None and  rect_in_imgR(crop_rect, img_R):
                # 裁剪图像
                x, y, w, h = crop_rect
                cropped_img = img_R[y:y+224, x:x+224]
                # 更新边界框坐标到裁剪后的坐标系
                bbox_R[:, 0] -= x
                bbox_R[:, 1] -= y
                return cropped_img, bbox_R, angle
    
    # 如果所有尝试都失败，返回None
    return None, None, None


def rotate(image, bbox, angle):
    """
    旋转图像和边界框，确保旋转后的图像包含所有原始内容
    
    Args:
        image: 原始图像
        bbox: 边界框 [x, y, w, h]
        angle: 旋转角度（角度制）
    
    Returns:
        rotated_img: 旋转后的图像
        rotated_bbox: 旋转后的边界框顶点坐标（相对于新图像）
    """
    h, w = image.shape[:2]
    
    # 计算旋转后需要的图像大小
    corners = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)
    
    # 转换角度为弧度
    angle_rad = np.deg2rad(angle)
    
    # 计算旋转中心（图像中心）
    cx, cy = w / 2, h / 2
    
    # 旋转后的角点
    rotated_corners = []
    for x, y in corners:
        # 平移到原点
        x -= cx
        y -= cy
        
        # 旋转
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # 平移回来
        x_rot += cx
        y_rot += cy
        
        rotated_corners.append([x_rot, y_rot])
    
    rotated_corners = np.array(rotated_corners)
    
    # 计算旋转后的边界
    x_min = np.floor(np.min(rotated_corners[:, 0]))
    y_min = np.floor(np.min(rotated_corners[:, 1]))
    x_max = np.ceil(np.max(rotated_corners[:, 0]))
    y_max = np.ceil(np.max(rotated_corners[:, 1]))
    
    # 计算新图像的大小
    new_width = int(x_max - x_min)
    new_height = int(y_max - y_min)
    
    # 创建新的转换矩阵，考虑平移
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # 添加平移，以确保所有内容都在新图像中
    M[0, 2] += -x_min
    M[1, 2] += -y_min
    
    # 执行仿射变换
    rotated_img = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_LINEAR)
    
    # 旋转边界框
    # x, y, width, height = bbox
    x, y, width, height = [round(val) for val in bbox]  # 确保是整数
    bbox_corners = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
    ])
    
    # 转换为齐次坐标
    bbox_corners_h = np.hstack((bbox_corners, np.ones((4, 1))))
    
    # 应用旋转加平移变换
    rotated_bbox_corners = np.dot(M, bbox_corners_h.T).T
    
    return rotated_img, rotated_bbox_corners


def get_min_rect(points):
    """
    计算点集的最小外接矩形
    """
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.intp)  # 使用 np.intp 替代 np.int0
    
    # 获取矩形的左上角和右下角坐标
    x_min = np.min(box[:, 0])
    y_min = np.min(box[:, 1])
    x_max = np.max(box[:, 0])
    y_max = np.max(box[:, 1])
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def rect_in_imgR(rect, image) -> bool:
    x, y, w, h = rect
    H, W, c = image.shape
    
    # 检查四个角的像素是否为空白
    corners = [
        (x, y),
        (x + w - 1, y),
        (x + w - 1, y + h - 1),
        (x, y + h - 1)
    ]
    for cx, cy in corners:
        if cx < 0 or cy < 0 or cx >= W or cy >= H:
            return False
        if np.all(image[cy, cx] == 0): 
            return False
    
    return True


def expand_rect(rect, image, angle, size=224):
    """
    在满足约束条件的情况下扩张矩形
    
    Args:
        rect: 初始矩形 [x, y, w, h]
        image: 图像
        size: 最小尺寸要求
    
    Returns:
        rect: 扩张后的矩形, 如果无法满足条件则返回None
    """
    h, w = image.shape[:2]
    rect_try = copy.deepcopy(rect)
    
    min_d, max_d = get_expand_range(rect_try, 'left', (h, w), angle, size)
    if max_d < min_d or max_d < 0:
        return None

    # 随机选择扩张距离
    expansion = np.random.uniform(min_d, max_d)  
    rect_try[0] -= expansion
    
    min_d, max_d = get_expand_range(rect_try, 'top', (h, w), angle, size)
    if max_d < min_d or max_d < 0:
        return None

    expansion = np.random.uniform(min_d, max_d)  
    rect_try[1] -= expansion
        
        
    # 成功
    rect_try[0] = math.floor(rect_try[0])
    rect_try[1] = math.floor(rect_try[1])
    rect_try[2] = size
    rect_try[3] = size
    return rect_try 


def get_expand_range(rect, edge, shape, angle, size: int=224) -> tuple:
    H, W = shape
    x, y, w, h = rect
    angle = (angle+180) % 90
    
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)
    tan = sin / cos
    
    if W < size or H < size: return (-1, -1)
    # 边S与图像边的交点距离公式为: dist = D / (sin_angle * cos_angle)
    # 其中D是边S与图像边界的垂直距离，D与大（向中间靠拢）dis也会逐渐增大
    # 需要保证 dist >= size
    # 所以 D = size * sin_angle * cos_angle
    D = size * sin * cos
    if H < 2*D + size or W < 2*D + size: return(-1, -1)
    
    # 考虑有没有包含边界框
    
    # [0-90] --- 其它角度都可以转到【0-90】来处理
    # 计算rect与imgR左边交点情况（左边）
    #   1. 两个交点位于imgR左上边：只考虑rect上交点
    #   2. rect上交点位于imgR左上边，rect下交点位于imgR左下边：都考虑
    #   3. 两个交点位于imgR左下边：只考虑rect下交点
    
    # 左边
    if edge == 'left':
        cor = W * sin  # imgR左边角点位于image上的 y坐标
        if y < cor and (y+h) < cor : # 情况1
            DL  = (cor-y) / tan
            
        elif y < cor and (y+h) > cor : # 情况2
            DL1 = (cor-y) / tan
            DL2 = (y+h-cor) * tan
            DL  = max(DL1, DL2)
            
        else: # y > cor and (y+h) > cor : # 情况3
            DL  = (y+h-cor) * tan
            
        max_d = min(x - max(DL, D), size-w)
        # 保证右边
        min_d = max(x + size + D - W, 0)
        
        return (min_d, max_d)
            
    # 上边
    elif edge == 'top':
        cor = W * cos  # imgR上边角点位于image上的 x坐标
        if x < cor and (x+w) < cor : # 情况1
            DL  = (cor-x) * tan
            
        if x < cor and (x+w) > cor : # 情况2
            DL1 = (cor-x) * tan
            DL2 = (x+w-cor) / tan
            DL  = max(DL1, DL2)
            
        if x > cor and (x+w) > cor : # 情况3
            DL  = (x+w-cor) / tan
            
        max_d = min(y - max(DL, D), size-h)
        
        # 保证下边
        min_d = max(y + size + D - H, 0)
        
        return (min_d, max_d)  
