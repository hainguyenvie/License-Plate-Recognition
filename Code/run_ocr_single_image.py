import psutil
import cv2
import time
from plate_and_text_detection import plate_and_text_detection
import pre_post_processing as processing
from image_process import prep_for_rec, batch_text_box
from ultralytics import YOLO
import openvino as ov
import ipywidgets as widgets
import pickle
from PIL import Image
import numpy as np
import re


def apply_contrast_enhancement(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
    This function handles both grayscale and BGR images.
    """
    try:
        if len(image.shape) == 2:  # Image is already grayscale
            gray = image
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Image is BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image format. The image must be either a 2D grayscale or a 3-channel BGR image.")

        print(f"Gray image shape: {gray.shape}")
        print(f"Gray image dtype: {gray.dtype}")

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # If the model expects a BGR image, convert grayscale back to BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        print(f"Enhanced image shape: {enhanced.shape}")
        return enhanced
    except Exception as e:
        print(f"Error during contrast enhancement: {e}")
        return None

def apply_skew_correction(image):
    """
    Correct skew in the image using edge detection and affine transformation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLines to find the most dominant line (which should align with the plate)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta - np.pi / 2) * 180 / np.pi  # Calculate angle relative to vertical
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image




def apply_thresholding(image):
    """
    Apply adaptive thresholding to binarize the image.
    Converts the image to grayscale if it's not already in that format.
    """
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # Image is already grayscale, no need to convert
        pass
    else:
        raise ValueError("Unsupported image format. Image must be either a 2D grayscale or 3-channel BGR image.")

    # Debugging: Print the shape and type of the image
    print(f"Grayscale image shape: {image.shape}")
    print(f"Grayscale image dtype: {image.dtype}")

    # Ensure image is of type uint8
    if image.dtype != 'uint8':
        image = image.astype('uint8')

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

    # Debugging: Print the shape of the thresholded image
    print(f"Thresholded image shape: {thresholded.shape}")

    return thresholded

def prepare_input_for_model(image):
    """
    Prepare the image for model input.
    Adjust dimensions, normalize, or batch the image as required by the model.
    """
    # Example: Add batch dimension if the model expects a batch
    if len(image.shape) == 2:  # Grayscale image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=0)  # Add channel dimension
    elif len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    else:
        raise ValueError("Unexpected image shape for model input preparation.")
    
    print(f"Prepared model input shape: {image.shape}")
    return image
def extract_char_and_numbers(text):
    # Regular expression to find a character and capture the last two digits before it
    pattern =   r'(\d{2})(\D)'
    
    # Search for the pattern in the string
    match = re.search(pattern, text)
    
    if match:
        # Extract the part before the matched pattern and add the result
        before_pattern = text[:match.start(1)]
        result = match.group(1) + match.group(2)  # Take last 2 digits and the character
        return result
    else:
        return text
def modify_string_based_on_digits(text):
    # Split the string by the dot
    parts = text.split('.')
    
    if len(parts) != 2:
        return text

    before_dot = parts[0]
    after_dot = parts[1]
    
    if len(before_dot) >= 4:
        before_dot = before_dot[-1:] 
    
    if len(after_dot) >= 3:
        after_dot = after_dot[:-1] 
    
    modified_text = before_dot + '.' + after_dot
    
    return modified_text


# Assuming you have the necessary imports for your models and processing utilities

def run_paddle_ocr_single_image_ver2(image_path, use_popup=False, txt_plate_det_model = '', rec_compiled_model = '', rec_output_layer = ''):
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

        # img = apply_thresholding(img)
        # img = prepare_input_for_model(img)

        # cv2.imshow("Prediction", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        scale = 1280 / max(img.shape)
        if scale < 1:
            img = cv2.resize(src=img, dsize=None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        # Preprocess the image for text detection.
      
        # frame, plate_coor = plate_det_yolo(plate_det_model, img)
        # print(plate_coor)
        # if not plate_coor:
        #     cv2.putText(img, str("Loi plate detection"), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        #     return img

   
        # if len(plate_coor) == 4:
        #     x1, y1, x2, y2 = plate_coor
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
            
        # yolo_plate_time = time.time()
        # print(f"Yolo plate detection time: {yolo_plate_time - start_time}")
        # # If the frame is larger than full HD, reduce size to improve the performance.

        txt_img, txt_boxx = plate_and_text_detection(txt_plate_det_model, img)

        # if not txt_boxx:
          
        #     cv2.putText(img, str("Loi text detection"), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        #     return img
        

        # x_offset, y_offset = plate_coor[0], plate_coor[1]
        # adjusted_boxes = [box + np.array([x_offset, y_offset]) for box in txt_boxx]

        

        # for box in adjusted_boxes:
        #     box = box.astype(int)  # Convert coordinates to integers
        #     cv2.polylines(img, [box], isClosed=True, color=(255, 0, 0), thickness=2)


        # yolo_text_time = time.time()
        # print(f"Yolo text detection time: {yolo_text_time - yolo_plate_time}")
     
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

        # if not txts:
        #     return ["Loi text recog"], plate_coor, txt_boxx, img

        
        print(txts)
        ocr_time = time.time()
        # print(f"OCR recognition time: {ocr_time - yolo_text_time}")
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
            if len(txts[1])> 3:
                txts[1] = extract_char_and_numbers(txts[1])
            if len(txts[0])>6:
                txts[0] = modify_string_based_on_digits(txts[0])                
            cv2.putText(txt_img, str(txts[1] + " " + txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        elif len(txts) == 2 and bool(re.search(r'[a-zA-Z]', str(txts[0]))):
            if len(txts[0])> 3:
                txts[0] = extract_char_and_numbers(txts[0])
            if len(txts[1])>6:
                txts[1] = modify_string_based_on_digits(txts[1])
            cv2.putText(txt_img, str(txts[0] + " " + txts[1]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        else:
            cv2.putText(txt_img, str(txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        
        # plate_text = ''
        # for t in txts:
        #     plate_text += t

        # cv2.putText(img, str(plate_text), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5) 
        

        # cv2.imshow("Prediction", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return txt_img
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











