from image_cv2 import pil_to_cv2
from ultralytics import YOLO
from PIL import Image
import cv2

def plate_det_yolo(model, image):
  
  pd = 20
  # image = Image.open(image)
  prediction = model.predict(source=image, conf=0.5, iou=0.5)
  

  # Convert PIL image to OpenCV format
  img_cv2 = pil_to_cv2(image)

  # Iterate over each prediction result
  for result in prediction:
      boxes = result.boxes.cpu().numpy()
      xyxy = boxes.xyxy

      # Draw bounding boxes on the predicted boxes
      # print(xyxy)
      if not xyxy.any():
          return [], ""

      for box in xyxy:
 
          x1, y1, x2, y2 = max(0, int(box[0]-pd)), max(0, int(box[1]-pd)), int(box[2]+pd), int(box[3]+pd)
          cropped_image = img_cv2[y1:y2, x1:x2]
          # print(box)
          # cv2.imwrite("../out.jpg", cropped_image)
          return cropped_image, [x1, y1, x2, y2]
 
      


# plate_det_model = YOLO('/home/os/techpro/Plate-Recognition/best_openvino_model_det_plate/best_openvino_model_newest')
# _, coors = plate_det_yolo(plate_det_model, "/home/os/techpro/Plate-Recognition/3.jpg")
# print(coors)