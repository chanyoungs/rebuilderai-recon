import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_cameras_labeled(json_file):
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    print(f"Loading '{json_file}'...")
    print(f"Found {len(data['frames'])} frames.")
    
    positions = []
    vecs_right = [] 
    vecs_up = []    
    vecs_view = []  
    labels = []

    scale = 0.3 # Size of the arrows

    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'])
        
        # 1. Position
        pos = c2w[:3, 3]
        positions.append(pos)
        
        # 2. Orientation Vectors
        vecs_right.append(c2w[:3, 0] * scale)      # Right (Red)
        vecs_up.append(c2w[:3, 1] * scale)         # Up (Green)
        vecs_view.append(-c2w[:3, 2] * scale)      # View (Blue) - note the negative Z

        # 3. Extract Label
        # Handle different path separators (/ or \) and get the last part
        path = frame.get('file_path', 'unknown')
        # Clean up path to get just the filename/folder name
        name = os.path.basename(os.path.normpath(path))
        # Remove extension if present (e.g., 'image.png' -> 'image')
        name = os.path.splitext(name)[0]
        labels.append(name)

    positions = np.array(positions)
    
    # Plot Dots
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='black', s=10)

    # Plot Arrows (RGB = XYZ)
    # Red = Right
    ax.quiver(positions[:,0], positions[:,1], positions[:,2], 
              [v[0] for v in vecs_right], [v[1] for v in vecs_right], [v[2] for v in vecs_right],
              color='r', linewidth=1)
    # Green = Up
    ax.quiver(positions[:,0], positions[:,1], positions[:,2], 
              [v[0] for v in vecs_up], [v[1] for v in vecs_up], [v[2] for v in vecs_up],
              color='g', linewidth=1)
    # Blue = View
    ax.quiver(positions[:,0], positions[:,1], positions[:,2], 
              [v[0] for v in vecs_view], [v[1] for v in vecs_view], [v[2] for v in vecs_view],
              color='b', linewidth=1)

    # --- ADD LABELS ---
    # We iterate and place text slightly offset from the camera center
    for i, txt in enumerate(labels):
        ax.text(
            positions[i, 0], 
            positions[i, 1], 
            positions[i, 2], 
            txt, 
            fontsize=8, 
            color='black'
        )

    # Plot Origin
    ax.scatter([0], [0], [0], c='magenta', marker='X', s=100, label='Origin')

    # Formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Labeled Cameras: {os.path.basename(json_file)}")
    
    # Aspect Ratio Fix
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'transforms.json'

    try:
        plot_cameras_labeled(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'")
    except Exception as e:
        print(f"Error: {e}")