import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from shapely.geometry import Polygon

######################################
#             for test
######################################

def rotated_rect(x_center, y_center, width, height, angle_degrees):
    """
    生成旋转矩形的四个顶点坐标
    :param x_center: 中心点x坐标
    :param y_center: 中心点y坐标
    :param width: 矩形宽度
    :param height: 矩形高度
    :param angle_degrees: 旋转角度（度数）
    :return: 旋转后的四个顶点坐标列表
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # 定义未旋转时的四个顶点相对坐标
    corners = np.array([
        [width / 2, height / 2],   # 右上
        [-width / 2, height / 2],  # 左上
        [-width / 2, -height / 2], # 左下
        [width / 2, -height / 2]   # 右下
    ])

    # 旋转并平移顶点
    rotated_corners = []
    for x, y in corners:
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        rotated_x = x_center + x_rot
        rotated_y = y_center + y_rot
        rotated_corners.append((rotated_x, rotated_y))
    
    return rotated_corners


def getIOU(true_param, pred_param, th=36, tw=36):
    """
    计算旋转矩形的交并比（IOU）
    :param true_param: 真实框参数 [x, y, scale, rotate]
    :param pred_param: 预测框参数 [x, y, scale, rotate]
    :param th: 模板高度（用于计算实际尺寸）
    :param tw: 模板宽度（用于计算实际尺寸）
    :return: IOU值
    """
    if len(pred_param) == 4:
        # 只有一个缩放参数
        pred_param = [pred_param[0], pred_param[1], pred_param[2], pred_param[2], pred_param[3]]
        true_param = [true_param[0], true_param[1], true_param[2], true_param[2], true_param[3]] 
    elif len(pred_param) == 3:
         # 没有缩放参数
        pred_param = [pred_param[0], pred_param[1],  1,  1, pred_param[2]]
        true_param = [true_param[0], true_param[1],  1,  1, true_param[2]] 
    
    true_param = np.array(true_param)
    pred_param = np.array(pred_param)
    
    # print(f"true_param: {true_param}")
    # print(f"pred_param: {pred_param}")
    # print(th, tw)
    
    # 解析参数
    true_x, true_y, true_sX, true_sY, true_r = true_param
    pred_x, pred_y, pred_sX, pred_sY, pred_r = pred_param

    # 计算实际宽高（假设scale为整体缩放因子）
    true_width = true_sX * tw
    true_height = true_sY * th
    pred_width = pred_sX * tw
    pred_height = pred_sY * th

    # 生成旋转矩形顶点
    true_poly = rotated_rect(true_x, true_y, true_width, true_height, -true_r)
    pred_poly = rotated_rect(pred_x, pred_y, pred_width, pred_height, -pred_r)

    # 创建多边形对象
    poly1 = Polygon(true_poly)
    poly2 = Polygon(pred_poly)

    # 检查多边形有效性
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # 计算交集和并集面积
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection

    return intersection / union if union != 0 else 0.0


def get_center(center, points, threshold=15):
    """
    基于固定阈值去除离群点的均值计算
    
    参数：
    center : 中心点坐标，形状为(2,)的数组
    points : 原始点集，形状为(n, 2)的数组
    threshold : 离群点判定阈值（欧氏距离），默认为15
    
    返回：
    mean : 去除离群点后的均值，形状为(2,)的数组
    """
    if len(points) == 0:
        return center
    
    # 计算所有点到中心点的欧氏距离
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    distances = np.sqrt(dx**2 + dy**2)
    
    # 过滤离群点
    mask = distances <= threshold
    filtered_points = points[mask]
    
    # 处理全为离群点的极端情况
    if len(filtered_points) == 0:
        return center  # 返回原始中心
    
    # 计算有效点的均值
    return np.mean(filtered_points, axis=0)


def draw_rotated_bbox(ax, param, color, th = 36, tw = 36, lineWidth=1):
    """
    在指定Axes上画旋转矩形和中心线。
    """
    if len(param) == 4:
        # 只有一个缩放参数
        param = [param[0], param[1], param[2], param[2], param[3]]
    elif len(param) == 3:
        # 没有缩放参数
        param = [param[0], param[1], 1, 1, param[2]]
    
    param = [float(p) for p in param]
    
    x, y, scale_x, scale_y, rotate = param
    h, w = th * scale_y, tw * scale_x

    # 创建矩形框
    rect = patches.Rectangle(
        (x - w/2, y - h/2),  # 左下角坐标
        w, h,                # 宽高
        linewidth=lineWidth,
        edgecolor=color,
        facecolor='none'
    )

    # 中心线
    line = plt.Line2D(
        [x, x + w/2],
        [y, y],
        color=color,
        linewidth=lineWidth,
        linestyle='-'
    )

    # 旋转变换
    transform = Affine2D().rotate_deg_around(x, y, -rotate) + ax.transData
    rect.set_transform(transform)
    line.set_transform(transform)

    ax.add_patch(rect)
    ax.add_line(line)
