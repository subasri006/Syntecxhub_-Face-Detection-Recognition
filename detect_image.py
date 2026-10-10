"""
Image Face Detection & Recognition Standalone Script
---------------------------------------------------
Detects and recognizes faces in static image files.

Usage:
  python detect_image.py --image path/to/image.jpg
  python detect_image.py --image path/to/image.jpg --output output/result.jpg --threshold 0.363
"""

import os
import argparse
import cv2
from face_engine import FaceEngine
from database import FaceDatabase


def process_image(image_path, output_path=None, threshold=0.363, show=True):
    """
    Process an image file: detect faces, recognize identities, draw annotations.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: '{image_path}'")
        return None

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Unable to load or decode image from '{image_path}'")
        return None

    print(f"[Info] Loaded image '{image_path}' ({img.shape[1]}x{img.shape[0]} px)")

    # Initialize Face Engine & Database
    try:
        engine = FaceEngine()
        db = FaceDatabase()
    except Exception as e:
        print(f"[Error] System initialization failed: {e}")
        return None

    # Detect faces
    num_faces, faces = engine.detect_faces(img)
    print(f"[Info] Total faces detected: {num_faces}")

    if num_faces == 0:
        print("[Info] No faces were found in the provided image.")
        annotated_img = img.copy()
    else:
        matches = []
        for i, face_det in enumerate(faces):
            aligned = engine.align_crop(img, face_det)
            embedding = engine.extract_embedding(aligned)
            identity, score = db.match_embedding(embedding, engine, threshold=threshold)
            matches.append((identity, score))
            print(f"  - Face #{i+1}: {identity} (Score: {score:.3f})")

        annotated_img = engine.draw_faces(img, faces, matches)

    # Save output if path specified
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, annotated_img)
        print(f"[Success] Processed image saved to '{output_path}'")

    # Display image window if enabled
    if show:
        window_name = f"Face Detection - {os.path.basename(image_path)}"
        # Resize window for comfortable preview if image is very large
        h, w = annotated_img.shape[:2]
        max_dim = 900
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            preview = cv2.resize(annotated_img, (int(w * scale), int(h * scale)))
        else:
            preview = annotated_img

        cv2.imshow(window_name, preview)
        print("[Info] Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return annotated_img


def main():
    parser = argparse.ArgumentParser(description="Detect and recognize faces in an image.")
    parser.add_argument("--image", "-i", required=True, help="Path to input image file.")
    parser.add_argument("--output", "-o", default=None, help="Path to save output annotated image.")
    parser.add_argument("--threshold", "-t", type=float, default=0.363, help="Recognition cosine similarity threshold (default: 0.363).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the image window.")

    args = parser.parse_args()

    # Default output path if not given and output dir exists
    output_path = args.output
    if not output_path and not args.no_show:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join("output", f"{base_name}_detected.jpg")

    process_image(args.image, output_path=output_path, threshold=args.threshold, show=not args.no_show)


if __name__ == "__main__":
    main()
