import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmcv.visualization import color_val
from torchvision import transforms

unloader = transforms.ToPILImage()


def imshow(img, gt_bboxes):
    image = img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    bbox_color = 'green'
    thickness = 3
    img = np.ascontiguousarray(image)
    bbox_color = color_val(bbox_color)
    gt_bboxes = gt_bboxes[0].cpu().numpy().tolist()
    for bbox in gt_bboxes:
        left_top = (int(bbox[0]), int(bbox[1]))
        right_bottom = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)

    plt.imshow(img)
    plt.show()
    # plt.pause(5)
