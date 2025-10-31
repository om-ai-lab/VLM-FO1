from detect_tools.upn import UPNWrapper
from PIL import Image
from vlm_fo1.mm_utils import draw_bboxes_and_save

upn_ckpt_path = "./resources/upn_large.pth"
upn_model = UPNWrapper(upn_ckpt_path)
img_pil = Image.open("demo/demo_image.jpg")
min_score = 0.3

fine_grained_proposals = upn_model.inference(img_pil)
fine_grained_filtered_proposals = upn_model.filter(
    fine_grained_proposals, min_score=min_score
    )

draw_bboxes_and_save(image=img_pil,
    detection_bboxes=fine_grained_filtered_proposals['original_xyxy_boxes'][0][:100], 
    output_path="demo/detection_result.jpg")