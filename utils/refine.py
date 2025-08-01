import cv2, math
import numpy as np


def rotate_template(template, angle, scale_x, scale_y):
    """旋转缩放模板，避免裁剪（warp 输出大小扩展为 sqrt(2)）"""
    th, tw = template.shape[:2]

    # 缩放尺寸
    scaled_th = int(th * scale_y)
    scaled_tw = int(tw * scale_x)
    scaled_template = cv2.resize(template, (scaled_tw, scaled_th))
    mask = np.ones((scaled_th, scaled_tw), dtype=np.uint8)

    # 计算放大后的安全边界尺寸（对角包围盒）
    diagonal_ratio = math.sqrt(2)
    safe_h = int(scaled_th * diagonal_ratio)
    safe_w = int(scaled_tw * diagonal_ratio)

    # 保证尺寸为偶数，便于后续中心对齐等操作
    safe_h += safe_h % 2
    safe_w += safe_w % 2

    # 新中心
    new_center = (safe_w // 2, safe_h // 2)

    # 计算仿射矩阵：从缩放后的模板中心 -> 新中心
    old_center = (scaled_tw // 2, scaled_th // 2)
    M = cv2.getRotationMatrix2D(old_center, angle, 1.0)
    M[0, 2] += new_center[0] - old_center[0]
    M[1, 2] += new_center[1] - old_center[1]

    rotated_template = cv2.warpAffine(scaled_template, M, (safe_w, safe_h), borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (safe_w, safe_h), borderValue=0)

    return rotated_mask, rotated_template


def extract_subpixel_patch(image, cx, cy, patch_w, patch_h):
    """从 image 中提取以 (cx, cy) 为中心的 patch，支持亚像素精度"""
    dst_center = (patch_w / 2, patch_h / 2)

    # 构造平移矩阵，把目标点平移到中心
    dx = dst_center[0] - cx
    dy = dst_center[1] - cy
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)

    patch = cv2.warpAffine(image, M, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return patch


def compute_masked_similarity(search_img, rotated_template, mask, cx, cy):
    """仅在模板对应区域中进行相似度计算"""
    
    patch = extract_subpixel_patch(search_img, cx, cy, rotated_template.shape[1], rotated_template.shape[0])

    if len(patch.shape) == 3:
        masked_patch = patch * mask[..., np.newaxis]
        patch_pixels = masked_patch[mask > 0].flatten().astype(np.float32)
        template_pixels = rotated_template[mask > 0].flatten().astype(np.float32)
    else:
        masked_patch = patch * mask
        patch_pixels = masked_patch[mask > 0].astype(np.float32)
        template_pixels = rotated_template[mask > 0].astype(np.float32)

    dot_product = np.dot(patch_pixels, template_pixels)
    norm1 = np.linalg.norm(patch_pixels)
    norm2 = np.linalg.norm(template_pixels)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def refine_angle(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle,
                         initial_range=15.0, step_size=1.5, imageShape=(224, 224), threshold=0.9):
    """自适应范围的角度细化"""
    
    def search_angles(angle_range, step):
        """在指定范围内搜索最佳角度"""
        angle_candidates = np.arange(
            pred_angle - angle_range, 
            pred_angle + angle_range + step, 
            step
        )
        
        best_score = -1
        best_angle = pred_angle
        
        for angle in angle_candidates:
            # 确保角度在[-180, 180]范围内
            normalized_angle = ((angle + 180) % 360) - 180
            
            mask, transformed_template = rotate_template(
                template, normalized_angle, 
                pred_scale_x, pred_scale_y
            )
            
            score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
            # score = compute_masked_similarity(search_img, transformed_template, mask)
            
            if score > best_score:
                best_score = score
                best_angle = normalized_angle
        
        return best_angle, best_score
    
    # 第一阶段：在初始范围内搜索
    best_angle, best_score = search_angles(initial_range, step_size)
    
    # 如果初始范围内找到满足阈值的结果，直接返回
    if best_score >= threshold:
        return best_angle, best_score
    
    # 第二阶段：扩大到全范围搜索
    # print(f"Initial search failed (score={best_score:.3f}), expanding to full range...")
    coarse_angle, coarse_score = search_angles(180.0, 10)
    if coarse_score > best_score:
        coarse_angle, coarse_score = search_angles(10, 1)
    
    # 如果粗搜索找到更好的结果
    if coarse_score > threshold:
        return coarse_angle, coarse_score
    
    # 如果全范围搜索也没有改善，返回原始预测
    return pred_angle, best_score


def refine_angle_bisection(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle,
                           initial_range=20.0, coarse_threshold=2.0, fine_step=0.5, imageShape=(224, 224), threshold=0.9):
    """使用二分法的角度细化（高效版）"""

    def get_score(angle):
        """计算给定角度下的匹配得分"""
        normalized_angle = ((angle + 180) % 360) - 180
        mask, transformed_template = rotate_template(
            template,normalized_angle,
            pred_scale_x, pred_scale_y
        )
        return compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)

    # 初始左右边界
    left = pred_angle - initial_range
    right = pred_angle + initial_range
    
    score_left = get_score(left)
    score_right = get_score(right)

    # 粗搜索阶段：二分
    while (right - left) > coarse_threshold:
        if score_left > score_right:
            right = (left + right) / 2
            score_right = get_score(right)
        else:
            left = (left + right) / 2
            score_left = get_score(left)

    # 精搜索阶段：细粒度遍历
    fine_angles = np.arange(left, right + fine_step, fine_step)
    best_score = -1
    best_angle = pred_angle

    for angle in fine_angles:
        score = get_score(angle)
        if score > best_score:
            best_score = score
            best_angle = angle
            
    if best_score > threshold:
        # 如果没有找到满足阈值的结果，返回原始预测
        return best_angle, best_score

    return refine_angle(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle, 
                 initial_range=initial_range*2, step_size=1, imageShape=imageShape, threshold=threshold)


def refine_scale_x(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, refined_angle,
                   search_range=0.21, step_size=0.03, threshold=0.92):
    """细化X方向的缩放 """
    
    # 生成候选缩放值
    scale_candidates = np.arange(
        pred_scale_x - search_range,
        pred_scale_x + search_range + step_size,
        step_size
    )
    
    # 限制缩放范围在合理区间内
    scale_candidates = scale_candidates[scale_candidates > 0.3]
    scale_candidates = scale_candidates[scale_candidates < 3.0]
    
    best_score = -1
    best_scale_x = pred_scale_x
    
    for scale_x in scale_candidates:
        mask, transformed_template = rotate_template(
            template, refined_angle, 
            scale_x, pred_scale_y
        )
        
        score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
        
        if score > best_score:
            best_score = score
            best_scale_x = scale_x
    
    # 如果改善不明显，返回原始值
    if best_score < threshold:
        return pred_scale_x, best_score
    
    return best_scale_x, best_score


def refine_scale_y(search_img, template, pred_x, pred_y, refined_scale_x, pred_scale_y, refined_angle,
                   search_range=0.15, step_size=0.015,  threshold=0.92):
    """细化Y方向的缩"""
    
    # 生成候选缩放值
    scale_candidates = np.arange(
        pred_scale_y - search_range,
        pred_scale_y + search_range + step_size,
        step_size
    )
    
    # 限制缩放范围在合理区间内
    scale_candidates = scale_candidates[scale_candidates > 0.3]
    scale_candidates = scale_candidates[scale_candidates < 3.0]
    
    best_score = -1
    best_scale_y = pred_scale_y
    
    for scale_y in scale_candidates:
        mask, transformed_template = rotate_template(
            template, refined_angle, 
            refined_scale_x, scale_y
        )
        
        score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
        
        if score > best_score:
            best_score = score
            best_scale_y = scale_y
    
    # 如果改善不明显，返回原始值
    if best_score < threshold:
        return pred_scale_y, best_score
    
    return best_scale_y, best_score


def refine_position(search_img, template, pred_x, pred_y, refined_scale_x, refined_scale_y, refined_angle,
                   search_range=0.5, step_size=0.1, threshold=0.95):
    """细化位置: x, y组合优化"""
    
    # 生成x和y方向的偏移量
    # 从-0.9到0.9，步长0.3: [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
    offsets = np.arange(-search_range, search_range + step_size, step_size)
    
    best_score = -1
    best_x = pred_x
    best_y = pred_y
    
    # 遍历所有位置组合
    for dx in offsets:
        for dy in offsets:
            # 计算新位置
            new_x = pred_x + dx
            new_y = pred_y + dy
            
            mask, transformed_template = rotate_template(
                template, refined_angle, 
                refined_scale_x, refined_scale_y
            )
            
            score = compute_masked_similarity(search_img, transformed_template, mask, new_x, new_y)
            
            if score > best_score:
                best_score = score
                best_x = new_x
                best_y = new_y
    
    # 如果改善不明显，返回原始值
    if best_score < threshold:
        return pred_x, pred_y, best_score
    
    return best_x, best_y, best_score
