import json
from pathlib import Path

import cv2
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
from tqdm import tqdm


class Dair2CoCo:

    def __init__(self, image_path, label_path, class2id=None):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.class2id = class2id
        self.all_info = []

    @staticmethod
    def parse(img_path, lb_path, class2id=dict(), save_ndarray=False):
        if save_ndarray:
            image = cv2.imread(str(img_path))
        else:
            image = img_path
        label = json.loads(lb_path.read_text())
        res = dict(image=image, label=[])

        for gt in label:
            category_name = gt['type']
            category_id = class2id[category_name]
            iscrowd = 0
            truncate = int(gt['truncated_state'])
            occlude = int(gt['occluded_state'])
            direction = 0
            x0y0wh = [
                float(gt['2d_box']['xmin']),
                float(gt['2d_box']['ymin']),
                float(gt['2d_box']['xmax']) - float(gt['2d_box']['xmin']),
                float(gt['2d_box']['ymax']) - float(gt['2d_box']['ymin'])
            ]
            if x0y0wh[2] <= 0 or x0y0wh[2] <= 0:
                print('Wrong')
                continue
            res['label'].append([
                None, x0y0wh, category_id, category_name, None, iscrowd,
                truncate, occlude, direction
            ])
        return res

    def read(self):
        par = tqdm(self.image_path.iterdir())
        for img_path in par:
            lb_path = self.label_path / (img_path.stem + '.json')
            self.all_info.append(self.parse(img_path, lb_path, self.class2id))

    def __call__(self, *args, **kwargs):
        self.read()
        return self.all_info


class Rope2CoCo:

    def __init__(self, image_path, label_path, class2id=None):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.class2id = class2id
        self.all_info = []

    @staticmethod
    def parse(img_path, lb_path, class2id=dict(), save_ndarray=False):
        if save_ndarray:
            image = cv2.imread(str(img_path))
        else:
            image = img_path
        label = lb_path.read_text().strip().splitlines()
        res = dict(image=image, label=[])

        for gt in label:
            gt = gt.split()
            category_name = gt[0].capitalize()
            if category_name == 'Unknown_unmovable':
                category_name = 'Trafficcone'
            elif category_name == 'Barrow':
                category_name = 'Barrowlist'
            elif category_name == 'Unknowns_movable':
                continue
            category_id = class2id[category_name]
            truncate = int(gt[1])
            occlude = int(gt[2])
            iscrowd = 0
            direction = 0
            x0y0wh = [
                float(gt[4]),
                float(gt[5]),
                float(gt[6]) - float(gt[4]),
                float(gt[7]) - float(gt[5])
            ]
            if x0y0wh[2] <= 0 or x0y0wh[2] <= 0:
                print('Wrong')
                continue
            res['label'].append([
                None, x0y0wh, category_id, category_name, None, iscrowd,
                truncate, occlude, direction
            ])
        return res

    def read(self):
        par = tqdm(self.image_path.iterdir())
        for img_path in par:
            lb_path = self.label_path / (img_path.stem + '.txt')
            self.all_info.append(self.parse(img_path, lb_path, self.class2id))

    def __call__(self, *args, **kwargs):
        self.read()
        return self.all_info


if __name__ == '__main__':

    roots = [
        '/media/ubuntu/NVME/DAIR_V2X/DAIR-V2X-I',
        '/media/ubuntu/NVME/DAIR_V2X/DAIR-V2X-C',
        '/media/ubuntu/NVME/DAIR_V2X/Rope3D',
        '/media/ubuntu/NVME/DAIR_V2X/Rope3D',
    ]
    image_dirs = [
        ['single-infrastructure-side-image'],
        ['cooperative-vehicle-infrastructure-infrastructure-side-image'],
        [
            'training-image_2a', 'training-image_2b', 'training-image_2c',
            'training-image_2d'
        ],
        ['validation-image_2'],
    ]
    label_dirs = [
        'single-infrastructure-side/label/camera',
        'cooperative-vehicle-infrastructure/infrastructure-side/label/camera',
        'training/label_2',
        'validation/label_2',
    ]

    coco_path = Path('/media/ubuntu/NVME/DAIR_V2X/COCO')
    coco_image_path = coco_path / 'train2017'
    coco_image_path.mkdir(parents=True, exist_ok=True)
    coco_label_path = coco_path / 'annotations' / 'instances_trainval2017.json'
    coco_label_path.parent.mkdir(parents=True, exist_ok=True)

    CLASSES = [Dair2CoCo, Dair2CoCo, Rope2CoCo, Rope2CoCo]

    class2id = {
        'Car': 0,
        'Truck': 1,
        'Van': 2,
        'Bus': 3,
        'Pedestrian': 4,
        'Cyclist': 5,
        'Tricyclist': 6,
        'Motorcyclist': 7,
        'Barrowlist': 8,
        'Trafficcone': 9,
    }
    id2class = dict(zip(class2id.values(), class2id.keys()))

    q = []
    for i, (root, image_dir, label_dir,
            CLASS) in enumerate(zip(roots, image_dirs, label_dirs, CLASSES)):
        _label_dir = Path(root) / label_dir
        for _image_dir in image_dir:
            _image_dir = Path(root) / _image_dir
            obj = CLASS(_image_dir, _label_dir, class2id)
            q.append(obj())

    cnt = 0
    coco = Coco()

    for name, id in class2id.items():
        coco.add_category(CocoCategory(id=id, name=name))

    for data in q:
        for kv in tqdm(data):
            i = f'{cnt:010d}.jpg'
            cnt += 1
            image = kv['image']
            label = kv['label']
            im = cv2.imread(str(image))
            st = coco_image_path / i
            cv2.imwrite(str(st), im)
            coco_image = CocoImage(
                file_name=i, height=im.shape[0], width=im.shape[1])
            flag = False
            for lb in label:
                coco_image.add_annotation(CocoAnnotation(*lb))
                flag = True
            if flag:
                coco.add_image(coco_image)
            else:
                del coco_image

    coco_json = coco.json
    save_json(coco_json, str(coco_label_path))
