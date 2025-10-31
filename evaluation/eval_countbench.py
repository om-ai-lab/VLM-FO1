import json
from tqdm import tqdm
import os
import re
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
)
import torch
import os



def eval_countbench(data_path, image_path, model_id, device):
    tokenizer, model, image_processors = load_pretrained_model(model_id, device=device)

    with open(data_path, "r") as f:
        data = json.load(f)
    
    gt_list = []
    pred_list = []
    for item in tqdm(data):
        question = item['question']
        gt = item['answer']
        bbox_list = item['bboxes']
        im = os.path.join(image_path, item['image'])

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": im},
                    },
                    {"type": "text", "text": question},
                ],
                "bbox_list": bbox_list,
            }
        ]

        generation_kwargs = prepare_inputs(model_id, model, image_processors, tokenizer, messages, device=device, max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False)
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()

        origin_ans = outputs
        
        ans = re.sub(r'<region\d+>', '', outputs)
        numbers = re.findall(r'(?<!region)\d+', ans)
        if numbers:
            pred = int(numbers[0])
        else:
            pred = 0
       
        pred_list.append(pred)
        gt_list.append(gt)
        if gt != pred:
            print(f"gt is {gt}, but pred is {origin_ans}")

        torch.cuda.empty_cache()

    correct = sum(1 for p, g in zip(pred_list, gt_list) if p == g)
    total = len(pred_list)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="evaluation/processed_data/countbench_with_upn_score_0.3_0.8.json")
    parser.add_argument("--image_path", type=str, default="data/CountBenchQA/images")
    # parser.add_argument("--data_path", type=str, default="evaluation/processed_data/pixmoCount_with_upn_score_0.3_0.8.json")
    # parser.add_argument("--image_path", type=str, default="data/pixmo_count/test_images")
    parser.add_argument("--model_id", type=str, default='resources/VLM-FO1_Qwen2.5-VL-3B-v01')   
    parser.add_argument("--device", type=str, default='cuda:0')                                                                                                  
    args = parser.parse_args()
    eval_countbench(args.data_path, args.image_path, args.model_id, args.device)