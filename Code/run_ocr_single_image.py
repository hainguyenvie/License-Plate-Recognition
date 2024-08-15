import psutil
import cv2
import time
from plate_det_yolo import plate_det_yolo
from text_det_yolo import text_det_yolo
import pre_post_processing as processing
from image_process import prep_for_rec, batch_text_box
from ultralytics import YOLO
import openvino as ov
import ipywidgets as widgets
import pickle
from PIL import Image
import numpy as np
import re



# Assuming you have the necessary imports for your models and processing utilities

def run_paddle_ocr_single_image_ver2(image_path, use_popup=False, plate_det_model = '', txt_det_model = '', rec_compiled_model = '', rec_output_layer = ''):
    """
    Perform PaddleOCR inference on a single image.

    Parameters:
        image_path: Path to the input image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
    """
    try:
        
        # process = psutil.Process()
        # initial_cpu = process.cpu_percent(interval=None)
        # initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        # Load the image.
        # Load the .pkl file
        # with open(image_path, 'rb') as file:
        #     data = pickle.load(file)
        #     print(data)

        # img = data.get('image')
        # img = image_path
        # img = Image.open(image_path)

        # print(img)

        start_time = time.time()
        img = cv2.imread(image_path)


        # cv2.imshow("Prediction", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        scale = 1280 / max(img.shape)
        if scale < 1:
            img = cv2.resize(src=img, dsize=None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        # Preprocess the image for text detection.
      
        frame, plate_coor = plate_det_yolo(plate_det_model, img)
        print(plate_coor)
        if not plate_coor:
            cv2.putText(img, str("Loi plate detection"), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            return img

   
        if len(plate_coor) == 4:
            x1, y1, x2, y2 = plate_coor
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
            
        yolo_plate_time = time.time()
        print(f"Yolo plate detection time: {yolo_plate_time - start_time}")
        # If the frame is larger than full HD, reduce size to improve the performance.

        txt_img, txt_boxx = text_det_yolo(txt_det_model, frame)

        if not txt_boxx:
          
            cv2.putText(img, str("Loi text detection"), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            return img
        

        x_offset, y_offset = plate_coor[0], plate_coor[1]
        adjusted_boxes = [box + np.array([x_offset, y_offset]) for box in txt_boxx]

        

        for box in adjusted_boxes:
            box = box.astype(int)  # Convert coordinates to integers
            cv2.polylines(img, [box], isClosed=True, color=(255, 0, 0), thickness=2)


        yolo_text_time = time.time()
        print(f"Yolo text detection time: {yolo_text_time - yolo_plate_time}")
     
        batch_num = 6
        img_crop_list, img_num, indices = prep_for_rec(txt_boxx, txt_img)

        # For storing recognition results, include two parts:
        # txts are the recognized text results, scores are the recognition confidence level.
        rec_res = [['', 0.0]] * img_num
        txts = []
        scores = []

        for beg_img_no in range(0, img_num, batch_num):

            # Recognition starts from here.
            norm_img_batch = batch_text_box(
                img_crop_list, img_num, indices, beg_img_no, batch_num)

            # Run inference for text recognition.
            rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

            # Postprocessing recognition results.
            postprocess_op = processing.build_post_process(processing.postprocess_params)
            rec_result = postprocess_op(rec_results)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if rec_res:
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

        if not txts:
            return ["Loi text recog"], plate_coor, txt_boxx, img

        
        print(txts)
        ocr_time = time.time()
        print(f"OCR recognition time: {ocr_time - yolo_text_time}")
        stop_time = time.time()
        processing_time_det = stop_time - start_time
        print(processing_time_det)

        # final_cpu = process.cpu_percent(interval=None)
        # final_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        # print(f"CPU Usage: {final_cpu}")
        # print(f"Memory Usage: {final_memory - initial_memory} MB")
        # if len(txts) == 2:
        #     txts = txts[::-1]
        
        # print(txts)

        if len(txts) == 2 and bool(re.search(r'[a-zA-Z]', str(txts[1]))):
            cv2.putText(img, str(txts[1] + " " + txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        elif len(txts) == 2 and bool(re.search(r'[a-zA-Z]', str(txts[0]))):
            cv2.putText(img, str(txts[0] + " " + txts[1]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        else:
            cv2.putText(img, str(txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

        # cv2.imshow("Prediction", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img
    except Exception as e:
        print(f"Error: {e}")


# core = ov.Core()

# device = widgets.Dropdown(
#     options=core.available_devices + ["AUTO"],
#     value='CPU',
#     description='Device:',
#     disabled=False,
# )

# rec_model_file_path = "../Text_reg/text_rec_new.xml"

# # Read the model and corresponding weights from a file.
# rec_model = core.read_model(model=rec_model_file_path)

# # Assign dynamic shapes to every input layer on the last dimension.
# for input_layer in rec_model.inputs:
#     input_shape = input_layer.partial_shape
#     print(input_shape)
#     input_shape[3] = -1
#     rec_model.reshape({input_layer: input_shape})

# rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")
# # Get input and output nodes.
# rec_input_layer = rec_compiled_model.input(0)
# rec_output_layer = rec_compiled_model.output(0)

# plate_det_model = YOLO('../best_openvino_model_det_plate/best_openvino_model_new')
# txt_det_model = YOLO('../best_openvino_model_text_det/best_openvino_model_new')

# image_path = '../13_5_10.jpg'

# txts, plate_coor, txt_boxes, frame = run_paddle_ocr_single_image_ver2(image_path, use_popup=False, plate_det_model = plate_det_model, txt_det_model = txt_det_model, rec_compiled_model= rec_compiled_model, rec_output_layer= rec_output_layer)
# print(txts)


# # frame = cv2.imread(image_path)
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

# cv2.imshow("Prediction", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()