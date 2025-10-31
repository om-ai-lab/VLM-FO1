OD_template = "Please detect {} in this image. Answer the question with object indexes."

OD_Counting_template = "How many {} are there in this image? Count each instance of the target object. Locate them with object indexes and then answer the question with the number of objects."

REC_template = "Please detect {} in this image. Answer the question with object indexes."

Region_OCR_template = "Please provide the ocr results of {} in the image."

Brief_Region_Caption_template = "Provide a brief description for {}."

Detailed_Region_Caption_template = "Provide a detailed description for {}."

Grounding_template = "Briefly describe this image and detect all mentioned objects. Answer with grounded object indexes."

Visual_Prompt_OD_template = "Using the provided object {} as a reference, identify all other objects of the same category in this image. Respond with object indexes."

Viusal_Region_Reasoning_template = "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Please give a detailed reasoning process process and provide image regions that can help you answer the question better. {}"
