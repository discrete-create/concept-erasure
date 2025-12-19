from yolo_utils import process_image
import argparse
import cv2
import os
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Image Processing with Noise Application")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--target_classes", type=str, nargs='+', required=True, help="List of target classes to apply noise")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--noise_std", type=float, default=25.0, help="Standard deviation of the Gaussian noise")
    parser.add_argument("--output_json", type=str, default="output.json", help="Path to save detected object JSON")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output images")
    args = parser.parse_args()

    out_image, mask, classes_in_image = process_image(
        args.image_path,
        args.model_path,
        args.target_classes,
        args.conf_threshold,
        args.noise_std
    )
    output_image_path = os.path.join(args.output_dir, "output_image.png")
    output_mask_path = os.path.join(args.output_dir, "mask.png")
    cv2.imwrite(output_image_path, out_image)
    cv2.imwrite(output_mask_path, (mask * 255).astype('uint8'))
    final_dict={"output_image":output_image_path,
                "mask_image":output_mask_path,
                "detected_objects":list(classes_in_image)}
    with open(args.output_json, 'w') as f:
        json.dump(final_dict, f, indent=4)


    