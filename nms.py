import numpy as np
def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Each box is represented as (x1, y1, x2, y2).
    """

    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])

    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def nms(bboxes, iou_threshold=0.2):
    """
    Apply Non-Maximum Suppression (NMS) on a set of bounding boxes.
    Bounding boxes are represented as a numpy array with shape (n, 5),
    where n is the number of bounding boxes and each row represents (x1, y1, x2, y2, confidence).
    """

    # Sort the bounding boxes by confidence score in descending order
    sorted_indices = np.argsort(bboxes[:, 4])[::-1]
    sorted_bboxes = bboxes[sorted_indices]

    selected_indices = []

    while len(sorted_bboxes) > 0:
        current_box = sorted_bboxes[0]
        selected_indices.append(sorted_indices[0])

        remaining_indices = sorted_indices[1:]
        ious = np.array([calculate_iou(current_box[:4], bbox[:4]) for bbox in sorted_bboxes[1:]])

        # Find indices of boxes with IoU less than the threshold
        overlapping_indices = np.where(ious <= iou_threshold)[0]

        sorted_indices = remaining_indices[overlapping_indices]
        sorted_bboxes = sorted_bboxes[1:][overlapping_indices]

    selected_bboxes = bboxes[selected_indices]

    return selected_bboxes