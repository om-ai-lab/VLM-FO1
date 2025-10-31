import torch
from PIL import Image
from detect_tools.upn import UPNWrapper
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
)
from vlm_fo1.task_templates import OD_template

# Paths to required files
img_path = "demo/demo_image.jpg"             # Path to input image
model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'  # VLM FO1 model path

bbox_list = [[161.0, 11.0, 292.0, 127.0], [268.0, 61.0, 428.0, 226.0], [12.0, 100.0, 140.0, 227.0], [205.0, 188.0, 332.0, 320.0], [326.0, 202.0, 478.0, 357.0], [136.0, 106.0, 269.0, 233.0], [25.0, 206.0, 200.0, 383.0]]

# Prepare chat messages with vision input and bounding boxes
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": img_path},
            },
            {
                "type": "text",
                "text": OD_template.format("orange"),
            },
        ],
        "bbox_list": bbox_list,
    }
]

# Load vision-language model and tokenizer
tokenizer, model, image_processors = load_pretrained_model(model_path)

# Prepare input for model generation
generation_kwargs = prepare_inputs(
    model_path, model, image_processors, tokenizer, messages,
    max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
)

# Run inference and decode output
with torch.inference_mode():
    output_ids = model.generate(**generation_kwargs)
    outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()

# Convert output prediction (indexes) to bounding box coordinates
bboxes = extract_predictions_to_bboxes(outputs, bbox_list)

img_pil = Image.open(img_path).convert("RGB")
# Draw detected bounding boxes and save visualization
draw_bboxes_and_save(
    image=img_pil,
    fo1_bboxes=bboxes,
    output_path="demo/vlm_fo1_result.jpg"
)