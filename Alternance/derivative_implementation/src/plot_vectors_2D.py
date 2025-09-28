#!/usr/bin/env python3

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

def read_array2d_from_binary(filename, dtype='d'):
    """
    Reads a binary file created by the C++ Array2D save function.

    The file format is expected to be:
    - Nx (size_t, 8 bytes)
    - Ny (size_t, 8 bytes)
    - Data (Nx * Ny * sizeof(dtype))

    Args:
        filename (str): Path to the binary file.
        dtype (str): Data type: 'i' for int32, 'f' for float32, 'd' for float64.

    Returns:
        np.ndarray: A 2D NumPy array, or None if an error occurs.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None

    dtype_map = {'i': np.int32, 'f': np.float32, 'd': np.float64}
    if dtype not in dtype_map:
        print(f"Error: Unsupported dtype '{dtype}'")
        return None
    
    try:
        with open(filename, 'rb') as f:
            # Read Nx and Ny (assuming C++ size_t is 8 bytes, standard on 64-bit)
            nx_bytes = f.read(8)
            ny_bytes = f.read(8)
            
            if len(nx_bytes) < 8 or len(ny_bytes) < 8:
                print(f"Error: File '{filename}' is too small to contain dimensions.")
                return None

            # Unpack bytes to integers. '<Q' means little-endian unsigned long long.
            nx = struct.unpack('<Q', nx_bytes)[0]
            ny = struct.unpack('<Q', ny_bytes)[0]
            
            # Read the rest of the file as the array data
            data = np.fromfile(f, dtype=dtype_map[dtype])
            
            if data.size != nx * ny:
                print(f"Warning: File '{filename}' data size {data.size} does not match dimensions read from file ({nx},{ny})")
                return None
            
            # Reshape the data to the correct 2D dimensions
            # The C++ code stores data in row-major order, which matches NumPy's default.
            return data.reshape((nx, ny))
            
    except Exception as e:
        print(f"Error reading or processing file '{filename}': {e}")
        return None

def plot_2d_files(files, dtype='d'):
    """
    Reads and plots multiple binary 2D arrays in a subplot grid.
    """
    n = len(files)
    if n == 0:
        print("No files to plot.")
        return
        
    # Set up the subplot grid layout
    if n == 2:
        rows, cols = 1, 2
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    axs = axs.flatten()
    
    successful_plots = 0
    for i, file in enumerate(files):
        arr = read_array2d_from_binary(file, dtype=dtype)
        
        if arr is not None:
            # Using contourf for smooth plots and .T to orient the data intuitively
            im = axs[i].contourf(arr.T, levels=50, cmap='viridis')
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(os.path.basename(file))
            axs[i].set_aspect('equal')
            successful_plots += 1
        else:
            # If a file fails to load, print a message and hide the empty plot
            print(f"Skipping plot for '{os.path.basename(file)}' due to read error.")
            axs[i].set_title(f"{os.path.basename(file)}\n(Failed to read)")
            axs[i].axis('off')
    
    # Hide any remaining unused subplots (e.g., if you plot 3 files on a 2x2 grid)
    for j in range(n, len(axs)):
        axs[j].axis('off')
    
    if successful_plots > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("No data was successfully plotted.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple 2D arrays from C++ binary files in subplots. "
                    "This script reads dimensions (Nx, Ny) directly from the files."
    )

    parser.add_argument('input_files', nargs='+', type=str, help="Binary files to plot")
    parser.add_argument('--dtype', choices=['i','f','d'], default='d', 
                        help="Data type of the array (d=double, f=float, i=int). Default is 'd'.")
    
    args = parser.parse_args()
    plot_2d_files(args.input_files, dtype=args.dtype)

if __name__ == "__main__":
    main()
