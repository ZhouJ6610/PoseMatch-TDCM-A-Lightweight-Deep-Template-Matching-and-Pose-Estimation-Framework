import csv
from pycocotools.coco import COCO
import math

def filter_data(coco_annotations_path, output_csv_path, tShape=(36, 36), scale_range=(0.5, 2)):
    """
    筛选 COCO 数据集中满足原始 w 和 h 缩放条件的目标（直接基于原始尺寸判断）
      
    参数:
        coco_annotations_path: COCO 数据集的注释文件路径（如 instances_train2017.json）
        output_csv_path: 输出 CSV 文件的路径
        target_shape: 模板的目标尺寸，默认为 (36, 36)
        scale_range: w 和 h 相对于 target_shape 的缩放范围，默认为 (0.5, 2)
    """
    th, tw = tShape
    count = 0
    
    # 加载 COCO 数据集
    coco = COCO(coco_annotations_path)
    img_ids = coco.getImgIds()
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # csv_writer.writerow(['img_name', 'x', 'y', 'w', 'h'])  # 写入表头
        
        # 遍历所有图像
        for img_id in img_ids:
            # 获取图像信息
            img_info = coco.loadImgs(img_id)[0]
            img_name = img_info['file_name']
            
            # 获取图像对应的标注
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # 遍历图像中的所有标注
            for ann in anns:
                # 获取边界框
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                x = math.floor(x)
                y = math.floor(y)
                w = math.ceil(w)
                h = math.ceil(h)
                
                # 计算 w 和 h 相对于目标尺寸的缩放比例
                scale_w = w / tw
                scale_h = h / th
                
                # 检查是否满足缩放条件
                if (scale_range[0] <= scale_w <= scale_range[1] and 
                    scale_range[0] <= scale_h <= scale_range[1]):
                    # 需要缩放为1，数据量不够，需进行裁剪
                    # w, h = tw, th
                    # 将满足条件的数据写入 CSV
                    csv_writer.writerow([
                        img_name,
                        x, y, w, h
                    ])
                    count += 1
    print(f"筛选完成，共找到 {count} 个满足条件的目标。")



if __name__ == "__main__":
    # COCO 数据集注释文件路径
    annotations_path = "/path-to/MS-CoCo/annotations/instances_train2017.json"
    
    # 输出 CSV 文件路径
    output_path = 'train.csv'
    
    filter_data(
        annotations_path, 
        output_path, 
        tShape=(36, 36), 
        scale_range=(0.5, 2)
    )
