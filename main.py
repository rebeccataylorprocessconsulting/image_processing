import cv2
import numpy as np
import os

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
BACKGROUND_COLOR = (255, 255, 255) # White background (BGR format)
PADDING_RATIO = 0.15                # 15% of the largest item dimension used as padding
MIN_AREA_THRESHOLD = 500            # Ignore contours smaller than this pixel area

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_multi_item_image(image, base_filename):
    h_img, w_img = image.shape[:2]

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to separate objects from background
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find external contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    item_count = 0

    for i, contour in enumerate(contours):
        # 1. Skip tiny noise contours
        if cv2.contourArea(contour) < MIN_AREA_THRESHOLD:
            continue
            
        item_count += 1
        x, y, w, h = cv2.boundingRect(contour)

        # 2. Create an isolated mask for JUST this specific item
        item_mask = np.zeros_like(thresh)
        cv2.drawContours(item_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 3. Apply mask to remove background specifically for this item
        white_bg = np.full(image.shape, BACKGROUND_COLOR, dtype=np.uint8)
        bg_removed_image = np.where(item_mask[:, :, None] == 255, image, white_bg)

        # 4. Calculate dynamic padding
        item_largest_dim = max(w, h)
        padding = int(item_largest_dim * PADDING_RATIO)
        
        padded_w = w + (padding * 2)
        padded_h = h + (padding * 2)
        square_size = max(padded_w, padded_h)
        
        # Center the square crop over the item center
        center_x = x + w // 2
        center_y = y + h // 2
        
        x_min = center_x - square_size // 2
        y_min = center_y - square_size // 2
        x_max = x_min + square_size
        y_max = y_min + square_size

        # 5. Canvas extraction (handles items near or touching image borders)
        canvas = np.full((square_size, square_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
        
        src_x_min = max(x_min, 0)
        src_y_min = max(y_min, 0)
        src_x_max = min(x_max, w_img)
        src_y_max = min(y_max, h_img)
        
        dst_x_min = src_x_min - x_min
        dst_y_min = src_y_min - y_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_max = dst_y_min + (src_y_max - src_y_min)
        
        canvas[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = bg_removed_image[src_y_min:src_y_max, src_x_min:src_x_max]

        # 6. Save the isolated item with a unique suffix
        name, ext = os.path.splitext(base_filename)
        output_filename = f"{name}_item{item_count}{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, canvas)
        
    print(f" -> Extracted {item_count} items from {base_filename}")


def process_folder():
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(INPUT_FOLDER, filename)

            image = cv2.imread(path)
            if image is None:
                continue

            print(f"Processing: {filename}")
            process_multi_item_image(image, filename)

if __name__ == "__main__":
    process_folder()