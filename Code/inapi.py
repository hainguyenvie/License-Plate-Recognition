from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
import ipywidgets as widgets
import openvino as ov
from ultralytics import YOLO
import cv2
from typing import Any

from run_ocr_single_image import run_paddle_ocr_single_image_ver2


# from main import plateRec

app = FastAPI()
core = ov.Core()

plate_det_model = YOLO('../best_openvino_model_det_plate/best_openvino_model_new')
txt_det_model = YOLO('../best_openvino_model_text_det/best_openvino_model_old')


device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='CPU',
    description='Device:',
    disabled=False,
)

rec_model_file_path = "../Text_reg/namhoai96.xml"

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



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import numpy as np
@app.post("/process_image")
async def process_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    # print(contents)
    image_data = pickle.loads(contents)
 
    results, plate_bbox, text_bbox, _ = run_paddle_ocr_single_image_ver2(
        image_data, 
        use_popup=False, 
        plate_det_model = plate_det_model, 
        txt_det_model = txt_det_model, 
        rec_compiled_model= rec_compiled_model, 
        rec_output_layer= rec_output_layer)
    
    
    plate_num = results[0] if len(results) == 1 else ((f"{results[0]} {results[1]}") if len(results[0]) < len(results[1]) else (f"{results[1]} {results[0]}"))


    return {"plate_number": plate_num,
            "plate_bbox": plate_bbox,
            # "text_bbox": text_bbox.tolist()
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
