#!/usr/bin/env python3
"""
Create test images for integration testing
"""

import os
import numpy as np
from PIL import Image

def create_test_images():
    """Create test images for integration testing"""
    test_dir = "test/resources"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create 9 test images with different characteristics
    for i in range(1, 10):
        img_path = os.path.join(test_dir, f"{i}.png")
        
        # Create 640x480 image (VGA resolution)
        width, height = 640, 480
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        if i == 1:  # Red solid
            img_array[:, :] = [255, 50, 50]
        elif i == 2:  # Green solid
            img_array[:, :] = [50, 255, 50]
        elif i == 3:  # Blue solid
            img_array[:, :] = [50, 50, 255]
        elif i == 4:  # Yellow pattern
            img_array[::2, ::2] = [255, 255, 0]  # Yellow squares
            img_array[1::2, 1::2] = [255, 255, 0]  # Yellow squares
        elif i == 5:  # Gradient
            for y in range(height):
                color_val = int(255 * y / height)
                img_array[y, :] = [color_val, color_val, 255 - color_val]
        elif i == 6:  # Checkerboard
            for y in range(height):
                for x in range(width):
                    if (x // 40 + y // 40) % 2 == 0:
                        img_array[y, x] = [255, 255, 255]
                    else:
                        img_array[y, x] = [0, 0, 0]
        elif i == 7:  # Rainbow stripes
            for x in range(width):
                color_phase = (x / width) * 6
                if color_phase < 1:  # Red to Yellow
                    img_array[:, x] = [255, int(255 * color_phase), 0]
                elif color_phase < 2:  # Yellow to Green
                    img_array[:, x] = [int(255 * (2 - color_phase)), 255, 0]
                elif color_phase < 3:  # Green to Cyan
                    img_array[:, x] = [0, 255, int(255 * (color_phase - 2))]
                elif color_phase < 4:  # Cyan to Blue
                    img_array[:, x] = [0, int(255 * (4 - color_phase)), 255]
                elif color_phase < 5:  # Blue to Magenta
                    img_array[:, x] = [int(255 * (color_phase - 4)), 0, 255]
                else:  # Magenta to Red
                    img_array[:, x] = [255, 0, int(255 * (6 - color_phase))]
        elif i == 8:  # Noise pattern
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        else:  # Mixed geometric
            # Draw some geometric shapes
            for y in range(height):
                for x in range(width):
                    if (x - width//2)**2 + (y - height//2)**2 < (min(width, height)//4)**2:
                        img_array[y, x] = [255, 165, 0]  # Orange circle
                    elif abs(x - width//2) < 20 or abs(y - height//2) < 20:
                        img_array[y, x] = [255, 255, 255]  # White cross
                    else:
                        img_array[y, x] = [100, 100, 200]  # Light blue background
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(img_path)
        print(f"Created test image: {img_path}")

if __name__ == "__main__":
    create_test_images()
    print("Test images created successfully!")