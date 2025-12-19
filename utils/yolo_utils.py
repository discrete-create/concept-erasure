import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path):
    model = YOLO(model_path)
    return model

def perform_inference(model, image):
    results = model(image)
    return results

def create_mask(boxes, image_shape, conf_threshold, target_classes, model_names):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    classes_in_image = set()
    for cls, conf, (x1, y1, x2, y2) in zip(
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.xyxy.cpu().numpy()):

        if conf < conf_threshold:
            continue

        if model_names[int(cls)] not in target_classes:
            continue
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        mask[y1:y2, x1:x2] = True
        classes_in_image.add(model_names[int(cls)])

    return mask,classes_in_image

def apply_noise_to_masked_regions(image, mask, noise_std):
    noise = np.random.normal(
        loc=127.5,
        scale=noise_std,
        size=image.shape
    )
    noise = np.clip(noise, 0, 255).astype(np.uint8)

    mask_3c = mask[:, :, None]
    out = image * (~mask_3c) + noise * mask_3c
    return out

def process_image(image_path, model_path, target_classes, conf_threshold, noise_std):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    results = perform_inference(model, img)
    boxes = results[0].boxes

    mask,classes_imag = create_mask(boxes, img.shape, conf_threshold, target_classes, model.names)
    out_image = apply_noise_to_masked_regions(img, mask, noise_std)

    return out_image,mask,classes_imag
"""
if __name__ == "__main__":
    # Example usage
    image_path = "/root/autodl-tmp/yolov12/屏幕截图 2023-12-12 204432.png"
    model_path = "/root/autodl-tmp/yolov12/yolov12s.pt"
    target_classes = ["person", "car"]
    conf_threshold = 0.5
    noise_std = 25.0

    out_image, mask, classes_in_image = process_image(
        image_path,
        model_path,
        target_classes,
        conf_threshold,
        noise_std
    )

    cv2.imwrite("output_image.png", out_image)
    cv2.imwrite("mask.png", (mask * 255).astype('uint8'))
"""
