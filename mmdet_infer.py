# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
from tqdm import trange

from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO

img_path = Path('/home/chenzhen/code/detection/datasets/dt_imgdata/coco_dt/val')
config_path = Path('/home/chenzhen/code/detection/mmdetection/configs/datang_detection/yolox_s_temp.py')
checkpoint = Path('/home/chenzhen/code/detection/mmdetection/checkpoint/800-v1.0-model.pth')
anno_path = Path('/home/chenzhen/code/detection/datasets/dt_imgdata/coco_dt/annotations/val.json')
draw_path = Path('/home/chenzhen/code/detection/datasets/dt_imgdata/v1.0-Pedestrian-imgs')
draw_path.mkdir(parents=True, exist_ok=True)

CLASSES = (
    "Car",
    "Bus",
    "Cycling",
    "Pedestrian",
    "driverless_Car",
    "Truck",
    "Animal",
    "Obstacle",
    "Special_Target",
    "Other_Objects",
    "Unmanned_riding"
)

red = (255, 0, 0)
blue = (0, 0, 255)
orange = (255, 165, 0)
red = red[::-1]
blue = blue[::-1]
orange = orange[::-1]


def box_iou(box1, box2, eps=1e-7):
    # box1 = box1[:4]
    # box2 = box2[:, :4]
    # a1, a2 = np.hsplit(box1, 2)
    # a1, a2 = np.expand_dims(a1, 1), np.expand_dims(a2, 1)
    # b1, b2 = np.hsplit(box2, 2)
    # b1, b2 = np.expand_dims(b1, 0), np.expand_dims(b2, 0),
    #
    # inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
    # iou = inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    # return iou

    bboxes1 = box1.astype(np.float32)
    bboxes2 = box2.astype(np.float32)
    cols = bboxes2.shape[0]
    ious = np.zeros((1, cols), dtype=np.float32)

    area1 = (bboxes1[2] - bboxes1[0] + eps) * (
            bboxes1[3] - bboxes1[1] + eps)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + eps) * (
            bboxes2[:, 3] - bboxes2[:, 1] + eps)
    for i in range(bboxes2.shape[0]):
        x_start = np.maximum(bboxes2[i, 0], bboxes1[0])
        y_start = np.maximum(bboxes2[i, 1], bboxes1[1])
        x_end = np.minimum(bboxes2[i, 2], bboxes1[2])
        y_end = np.minimum(bboxes2[i, 3], bboxes1[3])
        overlap = np.maximum(x_end - x_start + eps, 0) * np.maximum(
            y_end - y_start + eps, 0)
        union = area2[i] + area1 - overlap
        union = np.maximum(union, eps)
        ious = overlap / union
    return ious


def draw(image, bbox, name, color, xywh=True):
    if xywh:
        x0, y0, w, h = map(lambda x: int(round(x)), bbox)
        x1, y1 = x0 + w, y0 + h
    else:
        x0, y0, x1, y1 = map(lambda x: int(round(x)), bbox)
    cv2.rectangle(image, [x0, y0], [x1, y1], color, 2)
    cv2.putText(image, name, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, thickness=1)
    return image


def main():
    # build the model from a config file and a checkpoint file
    model = init_detector(str(config_path), str(checkpoint), device='cuda:0')
    example_coco = COCO(str(anno_path))
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    idx = category_names.index('Pedestrian')
    image_ids = example_coco.getImgIds(catIds=[])
    cate_id = example_coco.getCatIds('Pedestrian')
    # test a single image
    for i in trange(len(image_ids)):
        image_data = example_coco.loadImgs(image_ids[i])[0]
        path = img_path / image_data['file_name']
        try:
            img = cv2.imread(str(path))
        except Exception:
            continue
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotation_ids = example_coco.getAnnIds(
            imgIds=image_data['id'], catIds=cate_id, iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)

        result = inference_detector(model, img)
        # draw gt
        for anno in annotations:
            bbox = anno['bbox']
            cate = category_names[anno['category_id']]
            img = draw(img, bbox, cate, red)
        result = [i[i[:, 4] >= 0.1] for i in result]
        # all_result = np.concatenate(result)
        person = result.pop(idx)
        if not annotations and not person.size:
            continue
        elif annotations and not person.size:
            cv2.imwrite(str(draw_path / image_data['file_name']), img)
        else:
            others = np.concatenate(result)
            ious = box_iou(np.array(bbox), others)
            other_id = np.where(ious >= 0.3)
            for *bbox, s in person:
                img = draw(img, bbox, f'Pedestrian_{s:.3f}', blue, xywh=False)
            try:
                for *bbox, c in others[other_id, :]:
                    img = draw(img, bbox, category_names[int(c)], orange, xywh=False)
            except:
                pass

            cv2.imwrite(str(draw_path / image_data['file_name']), img)


if __name__ == '__main__':
    main()
