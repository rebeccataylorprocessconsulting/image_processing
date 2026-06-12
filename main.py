import cv2
import numpy as np
import os
from pathlib import Path
from rembg import remove, new_session
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from PIL import Image, ImageOps  # Added for EXIF orientation handling

# Automatically locate the user's system Downloads folder
DOWNLOADS_FOLDER = str(Path.home() / "Downloads")
OUTPUT_FOLDER = os.path.join(DOWNLOADS_FOLDER, "output_images")
BACKGROUND_COLOR = (255, 255, 255) # White background (BGR format)

# Initialize the AI session once
ai_session = new_session("u2net")

# Global variables for tracking state
start_time = 0.0
is_processing = False

def process_image_ai_only(image, base_filename):
    """
    Passes the image directly to the AI model for background removal
    and replaces the background with solid white.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Convert BGR to RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Run AI background removal
    rgba_out = remove(rgb_img, session=ai_session)

    # 3. Extract the Alpha mask
    alpha_mask = rgba_out[:, :, 3]

    # 4. Create a solid white background
    white_bg = np.full(image.shape, BACKGROUND_COLOR, dtype=np.uint8)

    # 5. Blend original image onto white background using the AI mask
    mask_normalized = alpha_mask[:, :, None] / 255.0
    bg_removed_image = (image * mask_normalized + white_bg * (1.0 - mask_normalized)).astype(np.uint8)

    # 6. Save the output to Downloads/output_images
    output_path = os.path.join(OUTPUT_FOLDER, base_filename)
    cv2.imwrite(output_path, bg_removed_image)

def update_stopwatch():
    """
    Background loop that updates the stopwatch label every 100ms.
    """
    if is_processing:
        elapsed = time.time() - start_time
        timer_label.config(text=f"Time Elapsed: {elapsed:.1f}s")
        root.after(100, update_stopwatch)

def background_processing(input_folder, files_to_process):
    """
    Handles image processing on a separate thread to keep UI elements moving.
    """
    global is_processing, start_time
    
    success_count = 0
    total_files = len(files_to_process)
    
    # Initialize UI indicators and start timing
    start_time = time.time()
    is_processing = True
    update_stopwatch()
    
    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0
    
    for idx, filename in enumerate(files_to_process, start=1):
        status_label.config(text=f"Processing image {idx} of {total_files}...")
        
        path = os.path.join(input_folder, filename)
        
        try:
            # 1. Open image with PIL to read EXIF metadata
            with Image.open(path) as img:
                # 2. Rotate/flip image based on the EXIF orientation tag automatically
                img = ImageOps.exif_transpose(img)
                # 3. Convert PIL RGB format to OpenCV BGR format array
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            image = None
        
        if image is not None:
            process_image_ai_only(image, filename)
            success_count += 1
            
        progress_bar["value"] = idx

    # Stop timing operations
    is_processing = False
    total_elapsed = time.time() - start_time
    
    # Finalise UI presentation
    status_label.config(text="Done!")
    timer_label.config(text=f"Total Time: {total_elapsed:.1f}s")
    upload_btn.config(state=tk.NORMAL)
    
    messagebox.showinfo("Success", f"Processed {success_count} images successfully in {total_elapsed:.1f} seconds!\nSaved to: {OUTPUT_FOLDER}")

def select_and_process_folder():
    """
    Opens a dialog box for the user to select an entire folder,
    then automatically starts background processing.
    """
    input_folder = filedialog.askdirectory(title="Select Folder Containing Images")
    if not input_folder:
        return 
        
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    files_to_process = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(valid_extensions)
    ]
    
    if not files_to_process:
        messagebox.showwarning("No Images Found", "No valid .jpg, .jpeg, or .png images were found in that folder.")
        return

    # Disable button to prevent multi-clicking while running
    upload_btn.config(state=tk.DISABLED)
    
    # Spawn background thread to prevent GUI lockup
    threading.Thread(target=background_processing, args=(input_folder, files_to_process), daemon=True).start()

# --- Create the GUI window ---
root = tk.Tk()
root.title("AI Background Remover")
root.geometry("450x300")
root.resizable(False, False)

# Instruction Label
instruction_label = tk.Label(root, text="Select a folder to remove backgrounds from all images.\nOutputs will save to your Downloads folder.", font=("Arial", 11), pady=15)
instruction_label.pack()

# Upload Folder Button
upload_btn = tk.Button(root, text="Select Folder", command=select_and_process_folder, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_btn.pack()

# Status Label
status_label = tk.Label(root, text="Ready", font=("Arial", 10, "italic"), pady=10, fg="blue")
status_label.pack()

# Progress Bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate")
progress_bar.pack(pady=5)

# Stopwatch/Timer Label
timer_label = tk.Label(root, text="Time Elapsed: 0.0s", font=("Arial", 10, "bold"), fg="#333333", pady=5)
timer_label.pack()

if __name__ == "__main__":
    root.mainloop()
