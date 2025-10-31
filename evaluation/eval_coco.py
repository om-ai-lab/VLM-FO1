import json
import os
from tqdm import tqdm
import math
from vlm_fo1.mm_utils import extract_predictions_to_indexes
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
)


def eval_coco(model_id, eval_data_path, original_data_path, img_folder, out_dir=None, device='cuda:0'):
    print(f"Evaluating {model_id} on {eval_data_path}...")
    tokenizer, model, image_processors = load_pretrained_model(model_id, device=device)

    output_path = os.path.join(out_dir, model_id.split("/")[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_list = []
    with open(eval_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    print(len(data_list))

    original_data = json.load(open(original_data_path, 'r'))
    catName_to_catId = {item['name']: item['id'] for item in original_data['categories']}

    res_list = []

    filename = eval_data_path.split('/')[-1].replace('.jsonl', '')
    out_data_path = f'{output_path}/{filename}_predictions.json'

    for data in tqdm(data_list):
        id = data['id']
        image_path = os.path.join(img_folder, data['image'])
        bbox_list = data['bbox_list']
        score_list = data['score_list']
        query = data['conversations'][0]['value']
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url":image_path},
                    },
                    {"type": "text", "text": query},
                ],
                "bbox_list": bbox_list,
            }
        ]
        
        generation_kwargs = prepare_inputs(model_id, model, image_processors, tokenizer, messages,device=device, max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False)

        
        try:
            output_ids = model.generate(**generation_kwargs)
            ans = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
        except:
            print(f"Error: {id}")
            continue
        print('ans:', ans)

        prediction_dict = extract_predictions_to_indexes(ans)
        for k, v in prediction_dict.items():
            for box in v:
                current_bbox = bbox_list[box]
                current_score = score_list[box]
                if k in catName_to_catId:
                    res = {
                            "image_id": id,
                            "category_id": catName_to_catId[k],
                            "bbox": [
                                current_bbox[0],
                                current_bbox[1],
                                current_bbox[2] - current_bbox[0],
                                current_bbox[3] - current_bbox[1]
                            ],
                            "score": current_score
                    }
                    res_list.append(res)

    print(f"predictions saved to: {out_data_path}")
    json.dump(res_list, open(out_data_path, 'w'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='resources/VLM-FO1_Qwen2.5-VL-3B-v01')
    parser.add_argument("--eval_data_path", type=str, default='evaluation/processed_data/cocoVal2017_with_upn_score_0.3_0.8.jsonl')
    parser.add_argument("--original_data_path", type=str, default='evaluation/processed_data/instances_val2017.json')
    parser.add_argument("--img_folder", type=str, default='data/coco/val2017')
    parser.add_argument("--out_dir", type=str, default='./evaluation')
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    eval_coco(args.model_id, args.eval_data_path, args.original_data_path, args.img_folder, args.out_dir, args.device)