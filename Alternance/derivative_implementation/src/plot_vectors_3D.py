#!/usr/bin/env python3

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

def read_array3d_from_binary(filename, dtype='d'):
    """
    Reads a binary file created by a C++ Array3D save function.

    The file format is expected to be:
    - Nx (size_t, 8 bytes)
    - Ny (size_t, 8 bytes)
    - Nz (size_t, 8 bytes)
    - Data (Nx * Ny * Nz * sizeof(dtype))

    Args:
        filename (str): Path to the binary file.
        dtype (str): Data type: 'i' for int32, 'f' for float32, 'd' for float64.

    Returns:
        np.ndarray: A 3D NumPy array, or None if an error occurs.
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
            # Read Nx, Ny, and Nz (assuming C++ size_t is 8 bytes on 64-bit)
            nx_bytes = f.read(8)
            ny_bytes = f.read(8)
            nz_bytes = f.read(8)
            
            if len(nx_bytes) < 8 or len(ny_bytes) < 8 or len(nz_bytes) < 8:
                print(f"Error: File '{filename}' is too small to contain dimensions.")
                return None

            # Unpack bytes to integers. '<Q' means little-endian unsigned long long.
            nx = struct.unpack('<Q', nx_bytes)[0]
            ny = struct.unpack('<Q', ny_bytes)[0]
            nz = struct.unpack('<Q', nz_bytes)[0]
            
            # Read the rest of the file as the array data
            data = np.fromfile(f, dtype=dtype_map[dtype])
            
            if data.size != nx * ny * nz:
                print(f"Warning: File '{filename}' data size {data.size} does not match dimensions read from file ({nx}, {ny}, {nz})")
                return None
            
            # Reshape the data to the correct 3D dimensions
            # The C++ code should store data in row-major order to match NumPy's default.
            return data.reshape((nx, ny, nz))
            
    except Exception as e:
        print(f"Error reading or processing file '{filename}': {e}")
        return None

def plot_3d_files(files, dtype='d', slice_axis=2, slice_index=None):
    """
    Reads and plots a 2D slice of multiple binary 3D arrays in a subplot grid.
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
        arr = read_array3d_from_binary(file, dtype=dtype)
        
        if arr is not None:
            # Determine the slice to plot
            current_slice_index = slice_index
            if current_slice_index is None:
                # Default to the middle slice if not specified
                current_slice_index = arr.shape[slice_axis] // 2

            if not (0 <= current_slice_index < arr.shape[slice_axis]):
                 print(f"Error: Slice index {current_slice_index} is out of bounds for axis {slice_axis} with size {arr.shape[slice_axis]} in file '{file}'.")
                 axs[i].set_title(f"{os.path.basename(file)}\n(Slice out of bounds)")
                 axs[i].axis('off')
                 continue
            
            # Extract the 2D slice
            if slice_axis == 0:
                slice_2d = arr[current_slice_index, :, :]
                axis_labels = ('Y', 'Z')
                title_axis = 'X'
            elif slice_axis == 1:
                slice_2d = arr[:, current_slice_index, :]
                axis_labels = ('X', 'Z')
                title_axis = 'Y'
            else: # slice_axis == 2
                slice_2d = arr[:, :, current_slice_index]
                axis_labels = ('X', 'Y')
                title_axis = 'Z'

            # Using contourf for smooth plots and .T to orient the data intuitively
            im = axs[i].contourf(slice_2d, levels=50, cmap='viridis')
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(f"{os.path.basename(file)}\n(Slice {title_axis}={current_slice_index})")
            axs[i].set_xlabel(axis_labels[0])
            axs[i].set_ylabel(axis_labels[1])
            axs[i].set_aspect('equal')
            successful_plots += 1
        else:
            # If a file fails to load, print a message and hide the empty plot
            print(f"Skipping plot for '{os.path.basename(file)}' due to read error.")
            axs[i].set_title(f"{os.path.basename(file)}\n(Failed to read)")
            axs[i].axis('off')
    
    # Hide any remaining unused subplots
    for j in range(n, len(axs)):
        axs[j].axis('off')
    
    if successful_plots > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("No data was successfully plotted.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D slices of multiple 3D arrays from C++ binary files. "
                    "This script reads dimensions (Nx, Ny, Nz) directly from the files."
    )

    parser.add_argument('input_files', nargs='+', type=str, help="Binary files to plot")
    parser.add_argument('--dtype', choices=['i','f','d'], default='d', 
                        help="Data type of the array (d=double, f=float, i=int). Default is 'd'.")
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                        help="The axis to slice along (0=X, 1=Y, 2=Z). Default is 2 (Z-axis).")
    parser.add_argument('--slice', type=int, default=None,
                        help="The index of the slice to display. Defaults to the middle slice.")
    
    args = parser.parse_args()
    plot_3d_files(args.input_files, dtype=args.dtype, slice_axis=args.axis, slice_index=args.slice)

if __name__ == "__main__":
    main()
