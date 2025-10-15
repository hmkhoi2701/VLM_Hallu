import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
from tqdm import tqdm

# Add current directory to path
sys.path.append('models/LLaVA')

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

os.makedirs("data/generated_data", exist_ok=True)

disable_torch_init()

model_name = get_model_name_from_path("liuhaotian/llava-v1.5-7b")
print(f"Loading model: {model_name}")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b",
    None,
    model_name
    )

with open('data/test.jsonl','r') as f:
    lines = f.readlines()
    
for line in tqdm(lines):
    data = json.loads(line)
    path = '/home/user/khoihm/val2014/' + data['image']

    image = Image.open(path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]
    query = "Describe this image."

    qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    # Create conversation and format prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    # Generate response with attention
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=image_sizes,
            do_sample=False,
            output_scores = True,
            output_attentions = True,
            return_dict_in_generate = True,
            max_new_tokens=2048
        )
        
    description = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    with open(f"data/generated_data/llava_test.jsonl", "a") as f:
        f.write(json.dumps({"image_id": data['image_id'], "image": data['image'], "description": description,
                            "annotations": data['annotations']}) + "\n")