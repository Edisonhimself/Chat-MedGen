import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


from chatmedgen.datasets.datasets.vqa_datasets import TestEvalData


from chatmedgen.common.eval_utils import prepare_texts, init_model, eval_parser
from chatmedgen.conversation.conversation import CONV_VISION_chatmedgen
from chatmedgen.common.config import Config



def list_of_str(arg):
    return list(map(str, arg.split(',')))


def check_and_create_file_path(file_path):
    directory = os.path.dirname(file_path)  
    if not os.path.exists(directory):  
        os.makedirs(directory)  







parser = eval_parser()
args = parser.parse_args()
cfg = Config(args)
model, vis_processor = init_model(args)
conv_temp = CONV_VISION_chatmedgen.copy()
conv_temp.system = ""
model.eval()






prompt = "[vqa] Based on the medical image, respond to this question with a short answer: Which part of the body does this image belong to?"
image_path = "examples/image/test.jpg"

test_eval_data = [{"prompt": prompt, "image_path": image_path}]




data = TestEvalData(test_eval_data, vis_processor)
eval_dataloader = DataLoader(data, batch_size=1, shuffle=False)


for image, prompt in eval_dataloader:
    text = prepare_texts(prompt, conv_temp)  # warp the texts with conversation template
    answer = model.generate(image, text, max_new_tokens=25, do_sample=False)
    print(answer)





