import time
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from PIL import Image

model_xml = Path('text_rec_new').with_suffix('.xml')
if not model_xml.exists():
    ov_model = ov.convert_model("inference.pdmodel")
    ov.save_model(ov_model, str(model_xml))
else:
    print(f"{model_xml} already exists.")