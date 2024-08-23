import numpy as np
from image_cv2 import pil_to_cv2
import cv2
from ultralytics import YOLO
from PIL import Image


def contains_bbox(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # Calculate the center of bbox_b
    bx_center = (bx1 + bx2) / 2
    by_center = (by1 + by2) / 2

    # Check if the center of bbox_b is within bbox_a
    return ax1 <= bx_center <= ax2 and ay1 <= by_center <= ay2


def plate_and_text_detection(model, image_path):
    boxx = []
    class_ids = []
    pd = 0
    tpd = 2
    # flag = False
    # Open the image using PIL
    # image = Image.open(image_path)
    img_cv2 = cv2.imread(image_path)
    # image = image_path


    # img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # img_pil = Image.fromarray(img_rgb)


    # Run the model prediction
    prediction = model.predict(source=img_cv2, conf = 0.3, iou = 0.5)

    # Convert PIL image to OpenCV format
    img_cv2 = pil_to_cv2(img_cv2)

    # Iterate over each prediction result
    for result in prediction:
        boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        class_ids = boxes.cls

        print(len)
        # if not xyxy.any():s
        #     return img_cv2, flag
        
        # Draw bounding boxes on the predicted boxes
        for box, class_id in zip(xyxy, class_ids):
            # if class_id == 1:  # Check if class_id is 1
            #     # flag = True
            #     x1, y1, x2, y2 = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)
            #     ar = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
            #     # cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            #     boxx.append(ar)
            #     # cv2.imshow("Prediction", img_cv2)
            #     # cv2.waitKey(0)

            if class_id == 0:  # Check if class_id is 1
                # x1, y1, x2, y2 = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)

                # cropped_image = img_cv2[y1:y2, x1:x2]
       
                # cv2.imwrite("../out.jpg", cropped_image)

                count = 0
                # flag = True
                plate_coor = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)
                for box, class_id in zip(xyxy, class_ids):

                    txt_coor = max(0, int(box[0]-tpd)), max(0, int(box[1]-1)), int(box[2]+tpd), int(box[3]+1)

                    if class_id == 1 and contains_bbox(plate_coor, txt_coor):
                        x1, y1, x2, y2 = txt_coor
                        ar = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
                        cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                        boxx.append(ar)
                        count += 1
                        if count == 2:
                            return img_cv2, boxx

                print(len(boxx))
                return img_cv2, boxx

# # Usage example
txt_plate_det_model = YOLO('../text_plate_det/best_openvino_model')
frame = '../13_5_00000000069000200_2062.jpg'
img_with_boxes, boxes = plate_and_text_detection(txt_plate_det_model, frame)
print(boxes)


