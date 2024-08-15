import numpy as np
from pathlib import Path
import openvino as ov
from IPython import display
import ipywidgets as widgets
from ultralytics import YOLO
import psutil
import cv2
import os
from PIL import Image
from run_ocr_single_image import run_paddle_ocr_single_image_ver2
import re


core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='CPU',
    description='Device:',
    disabled=False,
)

rec_model_file_path = "../Text_reg/text_rec_newest.xml"

# Read the model and corresponding weights from a file.
rec_model = core.read_model(model=rec_model_file_path)

# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")
# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

def run_paddle_ocr_folder(input_folder, output_folder, rec_compiled_model = '', rec_output_layer = ''):
    true_texts = []
    predicted_texts = []
    plate_det_model = YOLO('../best_openvino_model_det_plate/best_openvino_model_newest')
    txt_det_model = YOLO('../best_openvino_model_text_det/best_openvino_model_new')

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)

      
        img = run_paddle_ocr_single_image_ver2(image_path, use_popup=False, plate_det_model = plate_det_model, txt_det_model = txt_det_model, rec_compiled_model= rec_compiled_model, rec_output_layer= rec_output_layer)
        # predicted_texts.append(txts)


        # print(txts)
        # if len(txts) == 2 and bool(re.search(r'[a-zA-Z]', str(txts[1]))):
        #     cv2.putText(frame, str(txts[1] + " " + txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        # elif len(txts) == 2 and bool(re.search(r'[a-zA-Z]', str(txts[0]))):
        #     cv2.putText(frame, str(txts[0] + " " + txts[1]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        # else:
        #     cv2.putText(frame, str(txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)


        # if len(plate_coor) == 4:
        #     x1, y1, x2, y2 = plate_coor
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)


        # x_offset, y_offset = plate_coor[0], plate_coor[1]
        # adjusted_boxes = [box + np.array([x_offset, y_offset]) for box in txt_boxes]
        

        # for box in adjusted_boxes:
        #     box = box.astype(int)  # Convert coordinates to integers
        #     cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)


        # Save thez modified frame to the output folder

     
        # cv2.imshow("Prediction", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output_image_path = os.path.join(output_folder, image_name)
        # try:
        cv2.imwrite(output_image_path, img)
        # except cv2.error as e:
        #     print(f"Failed to save '{image_name}': {e}")
           

    return true_texts, predicted_texts


input_folder = "../image2"
output_folder = "../Output_imgs"
true_texts, predicted_texts = run_paddle_ocr_folder(input_folder, output_folder, rec_compiled_model= rec_compiled_model, rec_output_layer= rec_output_layer)
