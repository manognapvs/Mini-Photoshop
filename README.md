# Mini-Photoshop

## Project Overview

The primary goal of this project is to implement fundamental image processing operations, enabling users to manipulate images through an intuitive graphical user interface (GUI). This interface features a series of drop-down menus, under which the core operations of the software are neatly organized.

### Core Functionalities

- **Opening and Displaying BMP Files**: Users can open and view BMP images within the application.
- **Grayscale Conversion**: Converts color images to grayscale, reducing the complexity of processing.
- **Ordered Dithering**: Implements ordered dithering to images, adding a unique style.
- **Auto Level Adjustment**: Automatically adjusts the levels of an image to enhance its appearance.
- **Entropy and Huffman Coding Analysis**: Calculates and displays the entropy and Huffman coding for images, useful in understanding image compression.

### Optional Image Manipulation Features

Additional capabilities such as rotating, flipping, adjusting brightness, and applying Gaussian blur effects are included to provide users with more creative control over their images.

## Technologies and Libraries

The project is developed using Python v3 and incorporates several key libraries to enhance its functionality:

- **Tkinter**: Python’s standard GUI library, used for creating the application’s interface.
- **Pillow (PIL Fork)**: Adds image processing capabilities, supporting various image file formats.
- **NumPy**: Essential for high-performance mathematical operations and handling multidimensional arrays.
- **SciPy.stats**: Provides statistical functions, used here for entropy calculation.
- **Collections.Counter**: A dict subclass for counting hashable objects, used in frequency analysis for Huffman coding.
- **Heapq**: A heap queue algorithm, also known as the priority queue algorithm.

