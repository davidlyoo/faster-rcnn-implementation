import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class VOCBboxDataset:
    """
    PASCAL VOC 데이터셋을 로드하여 이미지, 바운딩 박스, 라벨을 반환하는 클래스
    """
    VOC_BBOX_LABEL_NAMES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    )
    
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(
            data_dir, f'ImageSets/Main/{split}.txt')
        
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = self.VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """
        i번째 샘플을 반환 (이미지, 바운딩 박스, 라벨, 난이도 정보 포함)
        """
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        
        bbox, label, difficult = [], [], []

        for obj in anno.findall('object'):
            is_difficult = int(obj.find('difficult').text) == 1
            if not self.use_difficult and is_difficult:
                continue

            difficult.append(is_difficult)
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            label.append(self.label_names.index(obj.find('name').text.lower().strip()))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.uint8)

        # 이미지 로드
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label