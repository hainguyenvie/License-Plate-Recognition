import numpy as np
from image_cv2 import pil_to_cv2
import cv2
from ultralytics import YOLO
from PIL import Image
import os


def text_det_yolo(model, image_path):

    boxx = []
    pd = 0
    flag = False
    # Open the image using PIL
    # image = image_path
    image = Image.open(image_path)

    # Run the model prediction
    prediction = model.predict(source=image, conf=0.79, iou=0.5)

    # Convert PIL image to OpenCV format
    img_cv2 = pil_to_cv2(image)

    # Iterate over each prediction result
    for result in prediction:
        
        boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy

        if not xyxy.any():
            return img_cv2, flag
        # Draw bounding boxes on the predicted boxes
        for box in xyxy:
            flag = True

  

            x1, y1, x2, y2 = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)
            # ar = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
            
            # cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            # cv2.imshow("Prediction", img_cv2)
            # cv2.waitKey(0)

            cropped_image = img_cv2[y1:y2, x1:x2]
            # boxx.append(ar)
    

    # return img_cv2, boxx
    return cropped_image, flag


txt_det_model = YOLO('../best_openvino_model_text_det/best_openvino_model_new')
# frame = '../valid/0a27389d-d167-445b-a0b0-e9ae33252df8_0.jpg'
# text_det_yolo(txt_det_model, frame)


input_folder = "../valid"
output_folder = "../Output_imgs"

count = 0
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    img, flag = text_det_yolo(txt_det_model, image_path)
    if flag:
        count += 1
    
    print(count)

    output_image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_image_path, img)


