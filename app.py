import tkinter as tk
import numpy as np
import math
import imageio
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps,ImageEnhance,ImageFilter
from collections import Counter
from scipy.stats import entropy
from heapq import heappush, heappop, heapify


# Dithering matrices
# Traditional Bayer matrices with values from 0 to the max value in the matrix
BAYER_MATRIX_2X2 = np.array([
    [0, 2],
    [3, 1]
], dtype=np.uint8)

BAYER_MATRIX_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.uint8)

adjusted_image = None
image_label = None
image_display = None

# ************ Image Processing Functions ************ #
## Function to open a file ##
def open_file():
    # Open a file dialog to select an image file
    global original_image  # Declare 'original_image' as a global variable to modify it inside this function

    # Get the file path from the user through a dialog box, filtering for BMP files only
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    
    # Check if the user selected a file; if not, exit the function early
    if not file_path:
        return
    # Attempt to open and validate the selected image file
    try:
        temp_image = Image.open(file_path)  # Open the image file with PIL's Image.open method
        
        # Check if the image is in the correct format (RGB) and within the size limits (1024x768)
        # The check for 'RGB' mode ensures the image is in 24-bit color (8 bits per channel)
        if temp_image.mode != 'RGB' or temp_image.size[0] > 1024 or temp_image.size[1] > 768:
            # Show an error message if the image doesn't meet the criteria
            messagebox.showerror("Error", "Image must be 24-bit RGB and dimensions within 1024x768.")
            return
        # Convert the image to RGB format in case it's not already in that format and store it in 'original_image'
        original_image = temp_image.convert("RGB")
        
        # Display the loaded image in the GUI by calling 'display_image' with the image and target label
        display_image(original_image, image_label)
    
    except Exception as e:
        # Show an error message if the file couldn't be opened or processed
        messagebox.showerror("Error", "Failed to open the file.\n" + str(e))


# ************ Grayscale Conversion ************ #
# Function to convert an image to grayscale using the luminious method #
def manual_grayscale_conversion(image):
    # Get the size of the image
    width, height = image.size  # Retrieve width and height of the input image
    
    # Create a new image for the grayscale version
    grayscale_image = Image.new("L", (width, height))  # Create a new image in grayscale mode ("L" mode stands for "luminance")
    
    # Get the pixel data from the original image
    pixels = image.load()  # Load pixel data from the original image
    
    # Get the pixel data for the grayscale image
    gray_pixels = grayscale_image.load()  # Prepare to manipulate pixels in the new grayscale image
    
    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]  # Get the RGB components of the current pixel
            
            # Apply the luminosity method to calculate the grayscale value
            # The luminosity method is a weighted sum of the RGB values, better approximating human perception of color
            luminosity = int(0.21 * r + 0.72 * g + 0.07 * b)  # Calculate grayscale value based on luminosity
            
            # Set the calculated luminosity value for the current pixel in the grayscale image
            gray_pixels[x, y] = luminosity
    
    # Return the processed grayscale image
    return grayscale_image

def grayscale_image():
    # Check if an image has been loaded
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")  # Show an error message if no image is loaded
        return
    # Convert the original image to grayscale
    gray_image = manual_grayscale_conversion(original_image)  # Apply the manual grayscale conversion function
    
    # Display the original and grayscale images side by side for comparison
    display_images_side_by_side(original_image, gray_image)  # Use a function to display both images in the GUI

    currently_edited_image = gray_image

# ************ Ordered Dithering ************ #
# Function to apply ordered dithering using a Bayer matrix #
def ordered_dithering_with_bayer(image, bayer_matrix):
    """
    Apply ordered dithering using a Bayer matrix to an input image.
    
    Args:
        image: A PIL Image object that represents the input image.
               It is expected to be in grayscale for dithering to work correctly.
        bayer_matrix: A numpy array that represents the Bayer dithering matrix.
                      This matrix determines the pattern of dithering applied to the image.
    
    Returns:
        A PIL Image object that represents the dithered image.
    """
    # Convert the PIL image to a numpy array for processing.
    img_array = np.array(image)
    
    # Normalize the Bayer matrix so its values scale from 0 to 255,
    # which matches the grayscale color range of the image.
    norm_matrix = bayer_matrix / np.max(bayer_matrix) * 255
    
    # Create a threshold matrix by tiling the normalized Bayer matrix to cover the entire image.
    # This repeats the Bayer pattern across the image dimensions.
    threshold_matrix = np.tile(norm_matrix, (img_array.shape[0] // bayer_matrix.shape[0] + 1, img_array.shape[1] // bayer_matrix.shape[1] + 1))
    
    # Crop the threshold matrix to match the exact dimensions of the input image.
    threshold_matrix = threshold_matrix[:img_array.shape[0], :img_array.shape[1]]
    
    # Apply the dithering: For each pixel, compare it against the corresponding value in the threshold matrix.
    # If the pixel value is greater than the threshold, set it to white (255); otherwise, set it to black (0).
    dithered_img = np.where(img_array > threshold_matrix, 255, 0)
    
    # Convert the dithered image back into a PIL Image object and return it.
    return Image.fromarray(dithered_img.astype(np.uint8))


# Function to display the output of ordered dithering #
def apply_ordered_dithering_bayer():
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    
    gray_image = manual_grayscale_conversion(original_image)
    # Choose the Bayer matrix to use
    bayer_matrix = BAYER_MATRIX_4X4  # Or BAYER_MATRIX_2X2 for a coarser effect
    dithered_image = ordered_dithering_with_bayer(gray_image, bayer_matrix)
    display_images_side_by_side(gray_image, dithered_image)

# ************ Auto-Leveling ************ #
# Function to apply auto-leveling to an image #
def auto_level(image):
    # Convert the image to a numpy array for image manipulation
    np_image = np.array(image)
    
    # Calculate the minimum and maximum pixel values
    min_vals = np.percentile(np_image, 1, axis=(0, 1))  # Using the 1st percentile
    max_vals = np.percentile(np_image, 99, axis=(0, 1))  # Using the 99th percentile
    
    # Create an output array of the same shape as the image
    output = np.zeros_like(np_image)
    
    # Apply the auto-leveling algorithm (stretching the histogram)
    for c in range(3):  # Assuming an RGB image
        output[..., c] = np.clip((np_image[..., c] - min_vals[c]) * 255.0 / (max_vals[c] - min_vals[c]), 0, 255)
    
    # Convert the output array back to an Image object
    auto_leveled_image = Image.fromarray(output, mode='RGB')
    return auto_leveled_image

# Function to display the output of auto-leveling #
def apply_auto_level():
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    auto_leveled = auto_level(original_image)
    display_images_side_by_side(original_image, auto_leveled)

# ************ Huffman Coding ************ #
def calculate_entropy_and_huffman(image):
    # Convert to grayscale
    gray_image = manual_grayscale_conversion(image)
    np_gray = np.array(gray_image, dtype=np.uint8)
    
    # Calculate frequencies of each pixel intensity
    pixel_counts = Counter(np_gray.flatten())
    total_pixels = np_gray.size
    
    # Calculate probabilities
    probabilities = {k: v / total_pixels for k, v in pixel_counts.items()}
    
    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    
    # For a perfectly efficient Huffman coding, average code length equals the entropy
    avg_code_length = entropy
    
    return entropy, avg_code_length

# Function to display output of Huffman coding #
def display_huffman_info():
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    clear_images()
    display_image(original_image, image_label)
    entropy, avg_code_length = calculate_entropy_and_huffman(original_image)
    huffman_info_label.config(text=f'Entropy: {entropy:.2f}\nAverage Huffman Code Length: {avg_code_length:.2f}')


## Optional Features: Rotate Image , Brightness Adjustment , Gaussian Blur , Flip Image ###

#************ Flip Image ************#

class FlipDirectionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("200x100")
        self.title("Choose Flip Direction")
        self.result = None

        tk.Button(self, text="Horizontal", command=lambda: self.set_result('horizontal')).pack(side="left", expand=True)
        tk.Button(self, text="Vertical", command=lambda: self.set_result('vertical')).pack(side="right", expand=True)

        self.transient(parent)  # Dialog is associated with the parent window
        self.grab_set()  # Modal dialog
        self.wait_window()  # Wait for the dialog to be destroyed

    def set_result(self, result):
        self.result = result
        self.destroy()

def flip_image_np(image_np, axis='horizontal'):
    """
    Flips an image horizontally or vertically.

    Parameters:
    - image_np: NumPy array of the image.
    - axis: 'horizontal' for a horizontal flip, 'vertical' for a vertical flip.
    
    Returns:
    - Flipped image as a NumPy array.
    """
    if axis == 'horizontal':
        # Flip horizontally
        return np.flip(image_np, axis=1)
    else:
        # Flip vertically
        return np.flip(image_np, axis=0)

# Function to apply flip to the image #
def apply_flip():
    global original_image
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return

    # Open the custom dialog to get the flip direction
    dialog = FlipDirectionDialog(root)
    flip_direction = dialog.result

    if not flip_direction:  # Check if the user closed the dialog without selection
        return

    # Convert PIL Image to numpy array for processing
    image_np = np.array(original_image)

    # Apply flip based on the selected direction
    flipped_image_np = flip_image_np(image_np, axis=flip_direction)

    # Convert numpy array back to PIL Image
    flipped_image = Image.fromarray(flipped_image_np)

    # Display the flipped image
    display_images_side_by_side(original_image, flipped_image)

# *********** Image Rotation *********** #
    
# Function to rotate an image using nearest-neighbor interpolation #
def rotate_np(image_np, angle):
    # Determine the dimensions of the input image
    height, width = image_np.shape[:2]
    # Convert the rotation angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the dimensions of the new image that can fully contain the rotated original image
    new_width = int(abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad)))
    new_height = int(abs(height * np.cos(angle_rad)) + abs(width * np.sin(angle_rad)))

    # Initialize a new image array with zeros (black) with the calculated dimensions
    rotated_image = np.zeros((new_height, new_width, image_np.shape[2]), dtype=np.uint8)

    # Calculate the center points of the original and new images
    original_center = (width / 2, height / 2)
    new_center = (new_width / 2, new_height / 2)

    # Iterate over each pixel in the new image
    for y_new in range(new_height):
        for x_new in range(new_width):
            # Calculate the corresponding pixel coordinates in the original image
            # by applying the inverse rotation transformation
            x_old = (x_new - new_center[0]) * np.cos(-angle_rad) - (y_new - new_center[1]) * np.sin(-angle_rad) + original_center[0]
            y_old = (x_new - new_center[0]) * np.sin(-angle_rad) + (y_new - new_center[1]) * np.cos(-angle_rad) + original_center[1]

            # Round the coordinates to the nearest integer since pixel indices must be integers
            x_old, y_old = int(round(x_old)), int(round(y_old))

            # Copy the pixel value from the original image to the new image
            # only if the calculated original coordinates are within the bounds of the original image
            if 0 <= x_old < width and 0 <= y_old < height:
                rotated_image[y_new, x_new] = image_np[y_old, x_old]

    # Return the rotated image array
    return rotated_image

# Function to rotate an image using a GUI dialog #
def rotate_image_gui():
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return

    # Ask the user for the rotation angle
    angle = tk.simpledialog.askfloat("Rotate", "Enter the rotation angle (in degrees):", minvalue=-360, maxvalue=360)
    if angle is None:  # User cancelled the dialog
        return
    
    # Convert PIL Image to numpy array for processing
    image_np = np.array(original_image)
    
    # Perform the rotation using imageio (we will define this process shortly)
    rotated_image_np = rotate_np(image_np, angle)
    
    # Convert numpy array back to PIL Image
    rotated_image = Image.fromarray(rotated_image_np)
    
    # Display the rotated image
    display_images_side_by_side(original_image, rotated_image)

# *********** Brightness Adjustment *********** #

def adjust_brightness(image, brightness_factor):
    """
    Adjusts the brightness of an image.
    
    Parameters:
    - image: A PIL Image object.
    - brightness_factor: An integer, positive to increase and negative to decrease brightness.
    
    Returns:
    - A PIL Image object with adjusted brightness.
    """
    img_array = np.array(image, dtype=np.int16)  # Use int16 to prevent overflow
    img_array += brightness_factor  # Adjust brightness
    img_array = np.clip(img_array, 0, 255)  # Clip to valid range
    # Convert back to PIL Image and return
    return Image.fromarray(img_array.astype(np.uint8))

# Function to open a window for adjusting brightness #
def open_brightness_window():
    global original_image  # Access the global variable to use the currently loaded image
    if original_image is None:  # Check if an image is loaded
        messagebox.showerror("Error", "No image loaded.")  # Show an error if no image is loaded
        return  # Exit the function if there's no image to adjust
    
    # Create a new top-level window for the brightness adjustment controls
    brightness_window = tk.Toplevel()
    brightness_window.title("Adjust Brightness")  # Set the title of the new window
    
    # Define a slider update function to be called whenever the slider value changes
    def slider_update(value):
        # Convert slider value from string to int
        brightness_factor = int(value)  # The slider returns a string, so convert it to an integer
        
        # Adjust the brightness of the original image using the specified factor
        adjusted_image = adjust_brightness(original_image, brightness_factor)
        
        # Update the main application window to display the adjusted image
        update_image_display(adjusted_image)
    
    # Create a slider control for adjusting brightness
    brightness_slider = tk.Scale(brightness_window, from_=-100, to=100,
                                 orient='horizontal',  # Arrange the slider horizontally
                                 label='Brightness',  # Label the slider
                                 command=slider_update)  # Specify the function to call when the slider value changes
    brightness_slider.pack()  # Add the slider to the window

    # Create a reset button to reset the brightness adjustment
    reset_button = tk.Button(brightness_window, text='Reset', 
                             command=lambda: brightness_slider.set(0))  # Reset slider to position 0 when clicked
    reset_button.pack()  # Add the reset button to the window

def update_image_display(adjusted_image):
    global image_display, image_label
    clear_images()  # Clear all images first
    display_image = ImageTk.PhotoImage(adjusted_image)
    image_label.config(image=display_image)
    image_label.image = display_image  # Keep a reference!

# ************ Gaussian Blur ************ #
    
# Function to generate a 2D Gaussian kernel #
def generate_gaussian_kernel(size, sigma=1.0):
    """
    Generates a 2D Gaussian kernel.
    
    Parameters:
    - size: The size of the kernel (e.g., 3, 5, 7). It determines the width and height of the kernel, which will be size x size.
    - sigma: The standard deviation of the Gaussian distribution. Controls the spread of the blur; higher values result in a more significant blur.
    
    Returns:
    - A 2D numpy array representing the Gaussian kernel.
    """

    # np.fromfunction allows us to generate an array by applying a function to each index.
    # The lambda function defined here computes the Gaussian equation at each point (x, y),
    # creating a 2D Gaussian kernel. The center of the kernel is positioned at (size-1)/2, (size-1)/2,
    # ensuring the kernel is centered around zero with the given size.
    # The Gaussian equation used here is a simplified 2D Gaussian function without rotation.
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)  # The shape of the kernel is determined by the 'size' parameter.
    )

    # The kernel is normalized by dividing it by its sum. This step ensures that the sum of all
    # elements in the kernel is 1, maintaining the original image's brightness after applying the filter.
    # Normalization is crucial for avoiding changes in the image's overall brightness when the kernel is applied.
    return kernel / np.sum(kernel)


def apply_gaussian_blur(image_np, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian Blur to a numpy image array.
    
    Parameters:
    - image_np: A numpy array representing the input image. Expected to have dimensions [height, width, channels].
    - kernel_size: The size of the Gaussian kernel. Determines the extent of the blur.
    - sigma: The standard deviation of the Gaussian distribution used in the kernel. Controls the spread of the blur.
    
    Returns:
    - A numpy array of the same shape as `image_np`, representing the blurred image.
    """
    # Generate the Gaussian kernel using the provided kernel size and sigma value
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    
    # Calculate the padding size needed to keep the image dimensions constant after convolution
    # Padding size is half the kernel size, as the kernel is centered on each pixel
    pad_size = kernel_size // 2
    
    # Pad the input image array to prevent border effects during the convolution
    # Padding is added to the top, bottom, left, and right sides but not across the color channels
    padded_image = np.pad(image_np, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    
    # Prepare an empty array for the output image, maintaining the same dimensions and data type as the input
    output = np.zeros_like(image_np)
    
    # Convolution operation: Apply the Gaussian kernel to every pixel in the image
    for x in range(image_np.shape[1]):  # Iterate over every column
        for y in range(image_np.shape[0]):  # Iterate over every row
            for c in range(image_np.shape[2]):  # Iterate over every color channel
                # Multiply the kernel with the corresponding image patch and sum the values
                # This convolution step results in the blurred pixel value at (y, x)
                output[y, x, c] = np.sum(kernel * padded_image[y:y+kernel_size, x:x+kernel_size, c])
                
    # Return the blurred image array
    return output

# Function to apply Gaussian blur to the image #
def apply_gaussian_blur_to_image():
    global original_image
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    
    # Convert PIL Image to numpy array
    image_np = np.array(original_image)
    
    # Apply Gaussian Blur
    blurred_image_np = apply_gaussian_blur(image_np, kernel_size=49, sigma=5.0)
    
    # Convert numpy array back to PIL Image
    blurred_image = Image.fromarray(blurred_image_np)
    
    # Display the blurred image
    display_images_side_by_side(original_image,blurred_image)

#************ Sharpening Image ************#

def sharpen_image(image):
    """
    Sharpens an image using a simple convolution with a sharpening kernel.
    
    Args:
        image: A numpy array of the image to be sharpened.
        
    Returns:
        A numpy array representing the sharpened image.
    """
    # Define a simple sharpening kernel
    kernel = np.array([
        [0, -1, -0],
        [-1,  5, -1],
        [0, -1,  0]
    ])
    
    # Pad the image to handle the edges
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), 'edge')
    
    # Initialize the sharpened image
    sharpened_image = np.zeros_like(image)
    
    # Perform the convolution operation
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):  # Handle each color channel
                # Extract the current 3x3 region
                region = padded_image[y:y+3, x:x+3, c]
                # Apply the kernel to the region (element-wise multiplication followed by sum)
                sharpened_value = np.sum(region * kernel)
                # Assign the sharpened value, ensuring it's within the valid range [0, 255]
                sharpened_image[y, x, c] = np.clip(sharpened_value, 0, 255)
    
    return sharpened_image

def display_sharpened_image():
    global original_image
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    
    # Convert PIL Image to NumPy array for processing
    original_image_array = np.array(original_image)
    
    # Sharpen the image
    sharpened_image_array = sharpen_image(original_image_array)
    
    # Convert the NumPy array back to a PIL Image
    sharpened_image = Image.fromarray(sharpened_image_array)
    
    # Now display the original and sharpened images side by side
    # Make sure your display_images_side_by_side function can handle PIL Images or modify it accordingly
    display_images_side_by_side(original_image, sharpened_image)

#************ Error Handling on GUI ************#
    
# Function to clear all images upon shifting to new operation #
def clear_images():
    # Function to clear all images from labels
    for label in [image_label, output_label, huffman_info_label]:
        label.config(image='')
        label.image = None
    huffman_info_label.config(text="")    

# Function to display output and input images side by side #  
def display_images_side_by_side(img1, img2):
    clear_images()  # Clear all images first
    # Display img1
    photo1 = ImageTk.PhotoImage(img1)
    image_label.config(image=photo1)
    image_label.image = photo1  # Keep a reference to the image

    # Display img2
    photo2 = ImageTk.PhotoImage(img2)
    output_label.config(image=photo2)
    output_label.image = photo2  # Keep a reference to the image

# Function to display an image #
def display_image(img, target_label):
    clear_images()  # Clear all images first
    photo = ImageTk.PhotoImage(img)
    target_label.config(image=photo)
    target_label.image = photo  # Keep a reference to the image

 # ************ Exit Program ************ #
# Function to exit the program #
def exit_program():
    root.destroy()

# ************ GUI Setup ************ #
# Initialize the GUI application
root = tk.Tk()
root.title("Mini-Photoshop")  # Set the window title for the application

original_image = None  # Initialize a global variable to store the currently loaded image

# Create a menu bar
menu_bar = tk.Menu(root)  # Initialize a new menu bar

# Main File Menu
file_menu = tk.Menu(menu_bar, tearoff=0)  # Create the File menu with tearoff disabled
menu_bar.add_cascade(label="File", menu=file_menu)  # Add the File menu to the menu bar

# Adding normal operations to the File menu
file_menu.add_command(label="Open File", command=open_file)  # Add "Open File" option
file_menu.add_command(label="Grayscale", command=grayscale_image)  # Add "Grayscale" option
file_menu.add_command(label="Ordered Dithering", command=apply_ordered_dithering_bayer)  # Add "Ordered Dithering" option
file_menu.add_command(label="Auto Level", command=apply_auto_level)  # Add "Auto Level" option
file_menu.add_command(label="Huffman", command=display_huffman_info)  # Add "Huffman" option for showing Huffman encoding info

# Separator to distinguish between normal and optional features
file_menu.add_separator()  # Adds a visual separator in the menu

# Submenu for Optional Operations
optional_menu = tk.Menu(file_menu, tearoff=0)  # Create a submenu for optional features
file_menu.add_cascade(label="Optional Operations", menu=optional_menu)  # Add the submenu to the File menu

# Adding optional features to the Optional Operations submenu
optional_menu.add_command(label="Flip Image", command=apply_flip)  # Add "Flip Image" option
optional_menu.add_command(label="Rotate Image", command=rotate_image_gui)  # Add "Rotate Image" option
optional_menu.add_command(label="Adjust Brightness", command=open_brightness_window)  # Add "Adjust Brightness" option
optional_menu.add_command(label="Gaussian Blur", command=apply_gaussian_blur_to_image)
optional_menu.add_command(label="Sharpen Image", command=display_sharpened_image)  # Add "Gaussian Blur" option

# Adding an exit option at the end of the file menu
file_menu.add_command(label="Exit", command=exit_program)  # Add "Exit" option to close the application

root.config(menu=menu_bar)  # Configure the root window to display the menu bar

# Main frame for content
content_frame = tk.Frame(root)  # Create a main frame to contain content
content_frame.pack(expand=True, fill='both')  # Pack the frame to expand and fill the window

# Define columns for left and right images
content_frame.columnconfigure(0, weight=1)  # Configure the left column to expand equally
content_frame.columnconfigure(1, weight=1)  # Configure the right column to expand equally, for displaying images

# Label for displaying images
image_label = tk.Label(content_frame)  # Create a label for displaying the original image
image_label.grid(row=0, column=0, sticky="ns")  # Place the label in the left column

# Label for displaying the images after processing
output_label = tk.Label(content_frame)  # Create another label for displaying the processed output image
output_label.grid(row=0, column=1, sticky="ns")  # Place the label in the right column

# Label for displaying Huffman information
huffman_info_frame = tk.Frame(root)  # Create a frame for displaying Huffman encoding information
huffman_info_frame.pack(side="top", padx=5, pady=5)  # Pack the frame at the bottom of the window
huffman_info_label = tk.Label(huffman_info_frame, text="")  # Create a label for the Huffman info within the frame
huffman_info_label.pack(side="top", padx=5, pady=5)  # Pack the label within the frame

root.mainloop()  # Start the Tkinter event loop to run the application

# End of the program