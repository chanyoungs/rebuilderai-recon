import os
from PIL import Image

def process_images(multi_views_folder, masks_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the multi_views directory
    image_files = [f for f in os.listdir(multi_views_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        multi_view_path = os.path.join(multi_views_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {image_file}. Skipping.")
            continue

        try:
            # Open the multi_views image and the mask image
            multi_view_image = Image.open(multi_view_path).convert("RGBA")
            mask_image = Image.open(mask_path).convert("L") # Convert mask to grayscale

            # Resize the mask image to match the multi_view_image dimensions if they don't match
            if multi_view_image.size != mask_image.size:
                mask_image = mask_image.resize(multi_view_image.size, Image.NEAREST)
                print(f"Resized mask {image_file} from {mask_image.size} to {multi_view_image.size}")

            pixels = multi_view_image.load()
            mask_pixels = mask_image.load()

            width, height = multi_view_image.size

            for x in range(width):
                for y in range(height):
                    # Get the mask pixel value (0 for black, 255 for white)
                    mask_value = mask_pixels[x, y]

                    # If the mask pixel is black (0), set the alpha of the corresponding multi_view pixel to 0
                    if mask_value == 0:
                        r, g, b, a = pixels[x, y]
                        pixels[x, y] = (r, g, b, 0)
            
            # Save the modified multi_views image
            multi_view_image.save(output_path)
            print(f"Processed and saved {image_file} to {output_folder}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    multi_views_folder = "multi_views"
    masks_folder = "masks"
    output_folder = "multi_views_alpha"
    process_images(multi_views_folder, masks_folder, output_folder)
