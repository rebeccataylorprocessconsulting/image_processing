import cv2
import numpy as np
import os

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
BACKGROUND_COLOR = (255, 255, 255) # White background (BGR format)
PADDING_RATIO = 0.15                # 15% of the largest item dimension used as padding
CLOSE_UP_THRESHOLD = 0.85           # Skip cropping if object takes up >85% of image area

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def crop_and_remove_background(image):
    h_img, w_img = image.shape[:2]
    total_image_area = w_img * h_img

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to smooth noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to separate object from background
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image  # fallback if nothing found

    # Vectorised: combine all contour points to get a single bounding box
    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    
    # --- UP-CLOSE SHOT CHECK ---
    # Calculate how much area the item occupies relative to the image size
    bounding_box_area = w * h
    coverage_ratio = bounding_box_area / total_image_area
    
    if coverage_ratio >= CLOSE_UP_THRESHOLD:
        print(" -> Up-close shot detected. Skipping crop.")
        return image # Return unchanged original image
    # ----------------------------

    # Create a solid white image of the exact same size as the original
    white_bg = np.full(image.shape, BACKGROUND_COLOR, dtype=np.uint8)
    
    # Use the threshold mask to isolate the product
    # Where mask is 255, keep product. Where mask is 0, use white background.
    bg_removed_image = np.where(thresh[:, :, None] == 255, image, white_bg)
    
    # Calculate padding dynamically based on the larger dimension of the item
    item_largest_dim = max(w, h)
    padding = int(item_largest_dim * PADDING_RATIO)
    
    # Calculate dimensions with dynamic padding included
    padded_w = w + (padding * 2)
    padded_h = h + (padding * 2)
    
    # Force 1:1 aspect ratio by using the larger dimension
    square_size = max(padded_w, padded_h)
    
    # Find the center of the detected object
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate new bounding coordinates to center the square
    x_min = center_x - square_size // 2
    y_min = center_y - square_size // 2
    x_max = x_min + square_size
    y_max = y_min + square_size

    # Use the background-removed image for cropping
    if x_min < 0 or y_min < 0 or x_max > w_img or y_max > h_img:
        # Create a blank square canvas filled with the background color
        canvas = np.full((square_size, square_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
        
        # Calculate valid source coordinates inside the original image
        src_x_min = max(x_min, 0)
        src_y_min = max(y_min, 0)
        src_x_max = min(x_max, w_img)
        src_y_max = min(y_max, h_img)
        
        # Calculate destination coordinates inside the new canvas
        dst_x_min = src_x_min - x_min
        dst_y_min = src_y_min - y_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_max = dst_y_min + (src_y_max - src_y_min)
        
        # Paste the background-removed content onto the canvas
        canvas[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = bg_removed_image[src_y_min:src_y_max, src_x_min:src_x_max]
        return canvas
    else:
        # Direct clean slice if it fits entirely within the original image
        return bg_removed_image[y_min:y_max, x_min:x_max]


def process_folder():
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(INPUT_FOLDER, filename)

            image = cv2.imread(path)
            if image is None:
                continue

            print(f"Processing: {filename}")
            # Call the updated function
            processed_img = crop_and_remove_background(image)

            output_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(output_path, processed_img)

            print(f"Processed: {filename}")

if __name__ == "__main__":
    process_folder()