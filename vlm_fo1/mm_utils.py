from PIL import Image
from PIL import ImageDraw
from io import BytesIO
import base64
import re 
import torch
from transformers import StoppingCriteria
from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX
import requests
from vlm_fo1.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
    DEFAULT_REGION_TOKEN, 
    DEFAULT_REGION_FEATURE_TOKEN
)
import torch
from transformers import TextStreamer
import random
import re
from typing import List, Tuple
import io
import base64


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenizes prompts containing <image> or <image_0>... special tokens.

    If the prompt uses <image_0>, <image_1>, ..., each is replaced with a placeholder index (-200).
    If the prompt uses <image>, it is replaced with image_token_index.

    Args:
        prompt (str): The prompt potentially containing image tokens.
        tokenizer: The tokenizer object.
        image_token_index (int): Token id to use when encountering <image> token.
        return_tensors (Optional[str]): If 'pt', return a torch tensor.

    Returns:
        List[int] or torch.Tensor: The tokenized input with image token indices inserted appropriately.
    """
    if "<image_0>" in prompt:
        # Case: prompt contains indexed image tokens like <image_0>, <image_1>, etc.
        image_token_pattern = re.compile(r"<image_(\d+)>")
        prompt_chunks = re.split(r'<image_[0-9]+>', prompt)
        image_tags = image_token_pattern.findall(prompt)

        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(tokenizer(chunk).input_ids)
            if i < len(image_tags):
                # Insert placeholder where <image_n> token was.
                input_ids.append(-200)
    else:
        # Case: prompt contains plain <image> tokens.
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            # Helper function to insert a separator token between chunks.
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        # If first chunk starts with <bos> token, make sure to keep it only once.
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        # Insert image_token_index between chunks.
        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])
    # Optionally convert output to PyTorch tensor.
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

def tokenizer_image_region_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, region_token_index=DEFAULT_REGION_INDEX, return_tensors=None):
    """
    Tokenizes prompts containing both <image> and <regionfeat> delimiters, inserting specified token indices.

    Each <image> chunk is split, and within that chunk, <regionfeat> locations receive region_token_index.

    Args:
        prompt (str): The prompt with <image> and <regionfeat> delimiters.
        tokenizer: The tokenizer object.
        image_token_index (int): Insert this at <image> splits.
        region_token_index (int): Insert this at <regionfeat> splits.
        return_tensors (Optional[str]): If 'pt', return torch tensor.

    Returns:
        List[int] or torch.Tensor: The tokenized input with region/image tokens placed.
    """
    # Split by <image> tags first.
    image_chunks = prompt.split('<image>')
    
    prompt_chunks = []
    for chunk in image_chunks:
        # Split each image chunk by <regionfeat>.
        obj_chunks = chunk.split('<regionfeat>')
        # Tokenize each subchunk.
        token_chunks = [tokenizer(c).input_ids for c in obj_chunks]
        prompt_chunks.append(token_chunks)
    
    input_ids = []
    offset = 0

    # If first chunk starts with <bos> token, include only once.
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and len(prompt_chunks[0][0]) > 0 and prompt_chunks[0][0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0][0])
    
    # Stitch together all chunks with region/image tokens at appropriate locations.
    for i, chunk_group in enumerate(prompt_chunks):
        if len(chunk_group) > 0:
            input_ids.extend(chunk_group[0][offset:])
        for chunk in chunk_group[1:]:
            input_ids.append(region_token_index)
            input_ids.extend(chunk)
        # Insert <image> token except after the last image chunk.
        if i < len(prompt_chunks) - 1:
            input_ids.append(image_token_index)
    # Optionally convert to PyTorch tensor.
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Implements custom stopping criteria for generation based on keywords:
    If the generated output contains any of the keywords, generation stops.
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # Remove BOS if present except for single token
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # Track the generation start length
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Checks if a keyword exists in the latest generated output ids for a single batch element.
        """
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Checks for keywords in each batch item; stops when all have satisfied the keyword condition.
        """
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

def load_image(image_file):
    """
    Loads an image from a local path, base64 string, URL, or PIL.Image.

    If the input image is smaller than 28x28, it will be resized to at least that size.

    Args:
        image_file (str or PIL.Image.Image): Image source.

    Returns:
        PIL.Image.Image: Loaded image in RGB mode, at least 28x28 in size.
    """
    if isinstance(image_file, Image.Image):
        image = image_file
    # Case: load from URL
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    # Case: load from base64-encoded string
    elif image_file.startswith("data:image/"):
        image = image_file.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))
    else:
        # Case: load from local file path
        image = Image.open(image_file).convert("RGB")
    
    # Ensure minimum size 28x28
    if image.width < 28 or image.height < 28:
        image = image.resize((max(28, image.width), max(28, image.height)))
    return image

def image_to_base64(img_pil):
    """
    Encodes a PIL Image as JPEG in base64 format.

    Args:
        img_pil (PIL.Image.Image): Source image.

    Returns:
        str: base64-encoded JPEG image string.
    """
    with io.BytesIO() as buffer:
        img_pil.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def draw_bboxes_and_save(
    image: Image.Image,
    fo1_bboxes: dict = {},
    detection_bboxes: List[Tuple[int, int, int, int]] = [],
    output_path: str = 'output.jpg',
    color: str = 'red',
    total_color: str = 'green',
    width: int = 2
) -> None:
    """
    Draws bounding boxes (both ground-truth/proposed and detection) on a PIL image and saves result.

    Args:
        image (PIL.Image.Image): Input PIL image object.
        fo1_bboxes (dict): Label -> List[bbox] mapping for annotation bboxes.
        detection_bboxes (List[Tuple]): List of detection bounding boxes; each bbox is (x_min, y_min, x_max, y_max).
        output_path (str): Path to save the output image.
        color (str): Color for fo1_bboxes.
        total_color (str): Color for detection_bboxes.
        width (int): Rectangle outline width.

    Returns:
        None
    """
    draw = ImageDraw.Draw(image)
    
    # Draw detection boxes with `total_color`
    for bbox in detection_bboxes:
        if len(bbox) != 4:
            print(f"警告: 跳过格式不正确的边界框 {bbox}")
            continue
        shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
        draw.rectangle(shape, outline=total_color, width=width)

    # Draw annotated bboxes with labels and `color`
    for bbox_label, bbox_list in fo1_bboxes.items():
        for bbox in bbox_list:
            if len(bbox) != 4:
                print(f"警告: 跳过格式不正确的边界框 {bbox}")
                continue
            shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            draw.rectangle(shape, outline=color, width=width)
            draw.text((bbox[0], bbox[1]), bbox_label, fill=color)

    # Save output image (catching common IO exceptions).
    try:
        image.save(output_path)
        print(f"图像已成功保存在: {output_path}")
    except IOError as e:
        print(f"错误: 无法保存图像到 {output_path}。原因: {e}")

def adjust_bbox(bbox_list, original_h, original_w, resize_h, resize_w):
    """
    Adjusts bounding boxes from original image size to resized image size, compensating for scaling.

    Args:
        bbox_list (List[List[float]]): List of original boxes [x1, y1, x2, y2].
        original_h (int): Original image height.
        original_w (int): Original image width.
        resize_h (int): Resized image height.
        resize_w (int): Resized image width.

    Returns:
        List[List[float]]: Bounding boxes transformed to resized image coordinates.
    """
    output_list = []
    def adjust_bbox_range(bbox, width, height):
        # Ensure all coordinates are within the original image border.
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        return [x1, y1, x2, y2]

    for bbox in bbox_list:
        bbox = adjust_bbox_range(bbox, original_w, original_h)
        bbox[0] = bbox[0] * resize_w / original_w
        bbox[1] = bbox[1] * resize_h / original_h
        bbox[2] = bbox[2] * resize_w / original_w
        bbox[3] = bbox[3] * resize_h / original_h
        output_list.append(bbox)
    return output_list

def extract_predictions_to_bboxes(prediction: str, bbox_list):
    """
    Parse prediction string in the expected format and map each ground label
    to its corresponding bounding boxes using bbox_list.

    Args:
        prediction (str): Model output string with <ground>...<objects>... markup.
        bbox_list (List[List[float]]): Full list of predicted or reference bounding boxes.

    Returns:
        dict: label -> list of bboxes
    """
    label_to_indexes = {}
    label_to_bboxes = {}

    match_pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"
    matches = re.findall(match_pattern, prediction)

    for label_text, indexes in matches:
        label_text = label_text.strip()
        indexes_tags = re.findall(r"<region\d+>", indexes)
        region_indexes = set([int(index.split("<region")[-1].split(">")[0]) for index in indexes_tags])
        if label_text not in label_to_indexes:
            label_to_indexes[label_text] = region_indexes
        else:
            label_to_indexes[label_text] = label_to_indexes[label_text] | region_indexes

    for label, indexes in label_to_indexes.items():
        label_to_bboxes[label] = [bbox_list[index] for index in indexes]

    return label_to_bboxes

def extract_predictions_to_indexes(prediction: str):
    """
    Parse prediction string, returning label -> set-of-indexes mapping.

    Args:
        prediction (str): Model prediction output.

    Returns:
        dict: label -> set(int)
    """
    label_to_indexes = {}
    match_pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"
    matches = re.findall(match_pattern, prediction)

    for label_text, indexes in matches:
        label_text = label_text.strip()
        indexes_tags = re.findall(r"<region\d+>", indexes)
        region_indexes = set([int(index.split("<region")[-1].split(">")[0]) for index in indexes_tags])
        if label_text not in label_to_indexes:
            label_to_indexes[label_text] = region_indexes
        else:
            label_to_indexes[label_text] = label_to_indexes[label_text] | region_indexes

    return label_to_indexes

def resize_shortest_edge_images_and_bboxes(
    image_list: List[Image.Image],
    bbox_lists: List,
    candidate_sizes: List[int] = [], 
    max_size: int = 2048
    ):
    """
    Randomly selects a size for the shortest edge, and proportionally resizes both images and bounding boxes.

    The function maintains the image aspect ratio and ensures that the resized dimensions do not exceed the specified max_size.
    Bounding boxes are transformed accordingly.

    Args:
        image_list (List[Image.Image]): A list of PIL Image objects.
        bbox_lists (List[List[List[float]]]): A list of lists of bounding boxes per image.
        candidate_sizes (List[int]): Optional list of sizes to choose the target short edge from.
        max_size (int): Maximum allowed long edge after resizing.

    Returns:
        Tuple[List[Image.Image], List[List[List[float]]]]:
            ([resized_image1, ...], [bbox_list1, ...]) - Possibly shape will match original (see below)

    Raises:
        ValueError: on input list length mismatch or emptiness.
    """
    bbox_tensor = torch.tensor(bbox_lists)
    # Normalize input: wrap bbox_lists into list-of-list, if needed.
    if len(bbox_tensor.shape) == 2 and bbox_tensor.shape[1] == 4:
        bbox_lists = [bbox_lists]

    if not image_list or not bbox_lists:
        raise ValueError("Input lists cannot be empty.")
    if len(image_list) != len(bbox_lists):
        raise ValueError("The lengths of the image list and the bounding box list must be the same.")

    # Randomly select short edge size (if given candidate sizes)
    if len(candidate_sizes) > 0:
        target_size = random.choice(candidate_sizes)
    else:
        target_size = None

    resized_images = []
    transformed_bbox_lists = []

    # Process each image and its corresponding bbox list
    for img, bboxes in zip(image_list, bbox_lists):
        original_width, original_height = img.size

        # Determine scaling factor to bring short edge to target_size
        shortest_side = min(original_width, original_height)
        if target_size:
            scale = target_size / shortest_side
        else:
            scale = 1.0

        # Propose new height and width with this scale
        new_height, new_width = int(original_height * scale), int(original_width * scale)

        # If resulting long edge exceeds max_size, rescale down so that it fits.
        longest_side = max(new_height, new_width)
        if longest_side > max_size:
            scale = max_size / longest_side
            new_height, new_width = int(new_height * scale), int(new_width * scale)
        # Ensure images are at least 28x28 (model may expect it)
        new_width = max(28, new_width)
        new_height = max(28, new_height)

        # Resize image, using BICUBIC for quality if shape changes
        if new_width == original_width and new_height == original_height:
            resized_img = img
        else:
            resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        resized_images.append(resized_img)

        # Transform bounding boxes
        current_transformed_bboxes = []
        scale_ratio_x = new_width / original_width
        scale_ratio_y = new_height / original_height
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            new_x1 = x1 * scale_ratio_x
            new_y1 = y1 * scale_ratio_y
            new_x2 = x2 * scale_ratio_x
            new_y2 = y2 * scale_ratio_y
            current_transformed_bboxes.append([new_x1, new_y1, new_x2, new_y2])
        transformed_bbox_lists.append(current_transformed_bboxes)

    # If original input was a single image (not list), unpack.
    if len(bbox_tensor.shape) == 2 and bbox_tensor.shape[1] == 4:
        return resized_images, transformed_bbox_lists[0]
    else:
        return resized_images, transformed_bbox_lists

def make_message_context(tokenizer, message, chat_format="chatml"):
    """
    Given a message dict, construct the prompt, tokenized context tokens, image URLs, and bbox_list.

    Handles both standard string 'content' and multi-part (list) content, appropriately placing image/region tokens.

    Args:
        tokenizer: tokenizer object
        message (dict): Contains role, content, and optionally bbox_list.
        chat_format (str): Optionally select chat format (default 'chatml').

    Returns:
        tuple: (inp, context_tokens, image_urls, bbox_list)
    """
    image_urls = []
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        nl_tokens = tokenizer.encode("\n")
        role = message["role"]
        content = message["content"]
        bbox_list = message.get("bbox_list", None)

        if role == "system":
            inp = f"{im_start}{role}\n{content}{im_end}\n"
            context_tokens = tokenizer.encode(
                role, allowed_special=set()) + nl_tokens + tokenizer.encode(content, allowed_special=set())
            context_tokens = im_start_tokens + context_tokens + im_end_tokens

        if role == "user":
            if isinstance(content, str):
                # Plain string message
                inp = f"{im_start}{role}\n{content}{im_end}\n"
                context_tokens = tokenizer.encode(
                    role, allowed_special=set()) + nl_tokens + tokenizer.encode(content,
                                                                                allowed_special=set())
                context_tokens = im_start_tokens + context_tokens + im_end_tokens
            if isinstance(content, list):
                # Multi-part message (text and image_url parts, maybe region tokens)
                inp = f"{im_start}{role}\n"
                image_count = 1
                for message_part in content:
                    if message_part["type"] == "text":
                        inp += f"{message_part['text']}"

                    if message_part["type"] == "image_url":
                        # Insert special vision/image tokens, possibly region tokens
                        inp += DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN + '\n'
                        # If regions exist, add per-region special token.
                        if bbox_list and len(bbox_list) > 0:
                            for idx, bbox in enumerate(bbox_list):
                                inp += DEFAULT_REGION_TOKEN.replace('<i>', str(idx)) + DEFAULT_REGION_FEATURE_TOKEN
                            inp += '\n'

                        image_urls.append(message_part['image_url']['url'])
                        image_count += 1
                inp += f"{im_end}\n"

                # Choose tokenizer logic based on whether bbox (region) list exists
                if bbox_list and len(bbox_list) > 0:
                    context_tokens = tokenizer_image_region_token(inp, tokenizer)
                else:
                    context_tokens = tokenizer_image_token(inp, tokenizer, image_token_index=IMAGE_TOKEN_INDEX)
        return inp, context_tokens, image_urls, bbox_list

def prepare_inputs(model_name, model, image_processors, tokenizer, messages, device="cuda", max_tokens=512, top_p=1.0, temperature=0.0, do_sample=False):
    """
    Fully prepares keyword arguments for model.generate (and compatible API) from messages and model specs.

    Handles prompt assembly, tokenization, image loading/preprocessing, region support, streaming, etc.
    Supports specific tweak for Qwen2.5-VL style vision tokens.

    Args:
        model_name (str): Model identifier string.
        model: Model/config object.
        image_processors (tuple): (primary, auxiliary) image processors.
        tokenizer: Tokenizer object.
        messages (list): Multi-message input list (chat history).
        device (str): Target (usually 'cuda' or 'cpu').
        max_tokens, top_p, temperature, do_sample: Standard generation kwargs.

    Returns:
        dict: ready-to-use argument dict for model.generate().
    """
    # For Qwen2.5-VL, patch vision special tokens globally.
    if 'qwen2.5-vl' in model_name.lower() or 'qwen2_5_vl' in model_name.lower():
        global DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        DEFAULT_IM_START_TOKEN = "<|vision_start|>"
        DEFAULT_IM_END_TOKEN = "<|vision_end|>"

    primary_image_processor, auxiliary_image_processor = image_processors

    prompt = ""
    input_tokens = []
    image_urls = []
    # Compose prompt and accumulate all components from provided messages
    for message in messages:
        inp, context_tokens, image_urls, bbox_list = make_message_context(tokenizer, message)
        prompt += inp
        input_tokens.extend(context_tokens)

    # Ensure a system prompt at start, if not already present.
    if "system" not in prompt:
        system_content = "system\nYou are a helpful assistant."
        system_prompt = "<|im_start|>" + system_content + "<|im_end|>" + "\n"
        prompt = system_prompt + prompt
        system_tokens = [151644] + tokenizer(system_content).input_ids + [151645] + tokenizer("\n").input_ids
        input_tokens = system_tokens + input_tokens

    # Ensure prompt ends with assistant's turn.
    if not prompt.endswith("<|im_start|>assistant"):
        last_assistant_prompt = "<|im_start|>" + "assistant" + "\n"
        prompt += last_assistant_prompt
        # last_assistant_tokens = [6] + self.tokenizer("assistant\n").input_ids
        last_assistant_tokens = [151644] + tokenizer("assistant\n").input_ids
        input_tokens.extend(last_assistant_tokens)

    primary_images_tensor = None
    auxiliary_images_tensor = None
    primary_image_grid_thws = None
    if image_urls:
        # Load images, resize them, and update bbox_list downstream
        images = [load_image(i) for i in image_urls]
        # print('original images[0].size:', images[0].size)
        images, bbox_list = resize_shortest_edge_images_and_bboxes(images, bbox_list, max_size=2048)
        # print('resized images[0].size:', images[0].size)
    
        # When region-indexed tokens are enabled
        if getattr(model.config, 'mm_use_region_index_token', False):
            origin_image_size = [image.size for image in images]
            aux_images = images.copy()
            auxiliary_images_tensor = [auxiliary_image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0].to(device) for i in aux_images]

            if bbox_list and len(bbox_list) > 0:
                # Limit number of bbox (for computational constraints, etc.)
                bbox_list = bbox_list[:100]
                resize_h, resize_w = auxiliary_images_tensor[0].shape[-2:]
                original_w, original_h = origin_image_size[0]
                # Adjust bbox to match resized images (post pre-processing)
                bbox_list = adjust_bbox(bbox_list, original_h, original_w, resize_h, resize_w)
                bbox_list = [torch.tensor(bbox_list)]
            else:
                bbox_list = None
        else:
            auxiliary_images_tensor = None

    # Preprocess primary images for main vision model branch
    primary_images = []
    primary_image_grid_thws = []
    for im in images:
        processed_data = primary_image_processor.preprocess(im, videos=None, return_tensors="pt")
        image_i = processed_data['pixel_values']
        image_grid_thw_i = processed_data['image_grid_thw']
        primary_images.append(image_i)
        primary_image_grid_thws.append(image_grid_thw_i)
    primary_images_tensor = [image_i.to(device) for image_i in primary_images]

    # For Qwen-style, force specific end-token as stopping criterion
    if "qwen" in model_name.lower():
        input_ids = torch.tensor([input_tokens]).to(device)
        keywords = ["<|im_end|>"]

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Default: greedy decoding if temperature=0. Else: enable sampling.
    if temperature == 0.0:
        do_sample = False
    else:
        do_sample = True

    print("question:================\n", prompt, "\n=================")
    # print("input ids:========", input_ids, "========")
    generation_kwargs = dict(
        inputs=input_ids,
        images=primary_images_tensor,
        images_aux=auxiliary_images_tensor,
        image_grid_thws=primary_image_grid_thws,
        bbox_list=bbox_list,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_tokens,
        streamer=streamer,
        top_p=top_p,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        pad_token_id=tokenizer.pad_token_id
    )
    return generation_kwargs

