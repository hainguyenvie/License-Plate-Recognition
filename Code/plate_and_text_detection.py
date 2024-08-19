import numpy as np
from image_cv2 import pil_to_cv2
import cv2
from ultralytics import YOLO
from PIL import Image


def plate_and_text_detection(model, image_path):
    boxx = []
    class_ids = []
    pd = 3
    # flag = False
    
    # Open the image using PIL
    # image = Image.open(image_path)s
    image = image_path

    # Run the model prediction
    prediction = model.predict(source=image, conf=0.5, iou=0.5)

    # Convert PIL image to OpenCV format
    img_cv2 = pil_to_cv2(image)

    # Iterate over each prediction result
    for result in prediction:
        boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        class_ids = boxes.cls
        
        # if not xyxy.any():
        #     return img_cv2, flag
        
        # Draw bounding boxes on the predicted boxes
        for box, class_id in zip(xyxy, class_ids):
            if class_id == 1:  # Check if class_id is 1
                # flag = True
                x1, y1, x2, y2 = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)
                ar = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
                # cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                boxx.append(ar)
                # cv2.imshow("Prediction", img_cv2)
                # cv2.waitKey(0)
    return img_cv2, boxx

# # Usage example
# txt_plate_det_model = YOLO('/home/os/techpro/Plate-Recognition/text_plate_det/best_openvino_model')
# frame = '/home/os/techpro/Plate-Recognition/13_5_10.jpg'
# img_with_boxes, boxes = plate_and_text_detection(txt_plate_det_model, frame)
