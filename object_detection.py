import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import ImageDraw
import sys

def get_obj_det_model(local=False):
    # download or load the model from disk
    if local:
        torch.hub.set_dir('.')
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800).eval()

def crop_beers(image, model, threshold, GPU=True):
    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes, classes, labels, preds = find_bottles(image, model, detection_threshold=threshold, GPU=GPU)
    if len(boxes) > 0:
        image_cropped = image.crop(tuple(boxes[0]))  # crop image: select only relevant part of pic
        # todo correct als er 2 boxes zijn (nu pak degene met hoogste pred, boxes is al gesorteerd op pred)
    else:
        image_cropped = image
    # image = draw_boxes(boxes, classes, labels, image)
    return image_cropped, len(boxes)


def find_bottles(image, model, detection_threshold=0.8, GPU=True):
    coco_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    device = torch.device('cuda' if GPU else 'cpu')
    model.to(device)

    # define the torchvision image transforms
    transform = transforms.Compose([
        transforms.ToTensor()])

    # transform the image to tensor
    image = transform(image)
    image = image.unsqueeze(0)  # add a batch dimension
    if GPU:
        image = image.cuda()
    outputs = model(image) # get the predictions on the image

    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # which labels are bottles?
    bottles = []
    for i in pred_classes:
        bottles.append(i == 'bottle')
    bottles = np.array(bottles)

    boxes = pred_bboxes.astype(np.int32)

    # get boxes above the threshold score & which are bottles
    if len(pred_bboxes)>0:
        relevant_outputs = (pred_scores >= detection_threshold) & bottles
    else:
        relevant_outputs = False

    if sum(relevant_outputs) > 0:
        return boxes[relevant_outputs], \
               len(list(np.array(pred_classes)[relevant_outputs])), \
               outputs[0]['labels'][relevant_outputs], \
               pred_scores[relevant_outputs]
    else:
        return[[], 0, [], []]


def draw_boxes(image, boxes, highlight_nr = 0):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for box in boxes:
        upper_left = (box[0], box[1])
        bottom_right = (box[2], box[3])
        if (box==boxes[highlight_nr]).all():
            draw.rectangle([upper_left, bottom_right], outline='chartreuse', width=10)
        else:
            draw.rectangle([upper_left, bottom_right], outline='chartreuse', width=2)
    return image_copy

def boxes_to_points(boxes):
    points = []
    for box in boxes:
        coord_x = (box[0] + box[2]) / 2
        coord_y = (box[1] + box[3]) / 2
        points.append((coord_x, coord_y))
    return points


def draw_points(image, points, point_size=10):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    point = (338.0, 390.5)
    for point in points:
        draw.ellipse([(point[0]-(point_size/2), point[1]-(point_size/2)),
                      (point[0]+(point_size/2), point[1]+(point_size/2))],
                     fill='chartreuse', outline='chartreuse')
    return image_copy


def rotate_points(image, center_coords, rotation):
    if rotation not in (0, 90, 180, 270):
        sys.exit('Please fill in one of the following rotations: 0, 90, 180, 270')

    center_coords2 = []
    for i in range(len(center_coords)):
        if rotation == 0:
            return center_coords
        else:
            if rotation == 90:
                x = image.size[0] - center_coords[i][1]
                y = center_coords[i][0]
            if rotation == 180:
                x = image.size[0] - center_coords[i][0]
                y = image.size[1] - center_coords[i][1]
            if rotation == 270:
                x = center_coords[i][1]
                y = image.size[1] - center_coords[i][0]
            center_coords2.append((x, y))
            return center_coords2

