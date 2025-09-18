#!/usr/bin/env python3

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def read_binary_vector(filename, vector_type='i', size_t_bytes=8):
    """
    Reads a vector from a binary file.

    Args:
        filename (str): The path to the binary file.
        vector_type (str): The data type ('i' for int, 'f' for float, 'd' for double).
        size_t_bytes (int): The size of size_t in bytes (e.g., 8 for 64-bit).

    Returns:
        numpy.ndarray: The vector data as a NumPy array, or None if an error occurs.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None

    with open(filename, 'rb') as f:
        size_t_format = 'Q' if size_t_bytes == 8 else 'L'
        try:
            dimension_data = f.read(size_t_bytes)
            if len(dimension_data) < size_t_bytes:
                print(f"Error: Could not read vector dimension from '{filename}'. File is too small.")
                return None
            
            dimension = struct.unpack(size_t_format, dimension_data)[0]

            dtype_map = {'i': np.int32, 'f': np.float32, 'd': np.float64}
            vector_data = np.fromfile(f, dtype=dtype_map[vector_type], count=dimension)

            if vector_data.size != dimension:
                print(f"Warning: In '{filename}', expected {dimension} elements but found {vector_data.size}.")

            return vector_data

        except Exception as e:
            print(f"An error occurred while reading '{filename}': {e}")
            return None

def main():
    """
    Main function to parse arguments and plot vectors.
    """
    # Step 1: Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Plot one or more vectors from binary files onto a single graph.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'input_files', 
        metavar='FILE', 
        type=str, 
        nargs='+',  # '+' means 1 or more arguments
        help="One or more paths to the binary vector files to be plotted."
    )
    
    parser.add_argument(
        '--type', 
        type=str, 
        default='i', 
        choices=['i', 'f', 'd'],
        help="The data type of the vector elements:\n"
             "  'i' for int (default)\n"
             "  'f' for float\n"
             "  'd' for double"
    )

    parser.add_argument(
        '--size_t', 
        type=int, 
        default=8, 
        choices=[4, 8],
        help="The size of size_t in bytes (e.g., 8 for 64-bit, 4 for 32-bit). Default is 8."
    )
    
    args = parser.parse_args()

    # Step 2: Read data from all provided files
    vectors = []
    filenames = []
    for file_path in args.input_files:
        print(f"Reading data from '{file_path}'...")
        vector = read_binary_vector(file_path, vector_type=args.type, size_t_bytes=args.size_t)
        if vector is not None:
            vectors.append(vector)
            filenames.append(os.path.basename(file_path))

    if not vectors:
        print("No valid vector data was loaded. Aborting plot.")
        return

    # Step 3: Plot all the vectors on the same graph
    plt.figure(figsize=(12, 8))
    
    # Use a cycle of styles to ensure visual distinction
    colors = plt.cm.jet(np.linspace(0, 1, len(vectors)))
    markers = ['o', 'x', 's', '^', 'v', 'D', '*']
    
    for i, (vector, filename) in enumerate(zip(vectors, filenames)):
        plt.plot(
            vector, 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            color=colors[i], 
            label=f'Vector from {filename}'
        )

    # Add titles and labels for clarity
    plt.title(f"Comparison of {len(vectors)} Vectors")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
