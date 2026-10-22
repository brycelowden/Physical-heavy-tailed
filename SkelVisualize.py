import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pyvista as pv

# ==========================================
# USER DIRECTORY & STORAGE CONFIGURATIONS
# ==========================================
BASE_FILE_PATH = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireSkeletons/'
OUTPUT_DIR = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 1/Top 9/'

# ==========================================
# CAMERA CONFIGURATION
# ==========================================
# Set to None to run interactively, align the view, and capture camera coordinates.
# Once captured, copy the logged tuple here to render all 9 neurons from the exact same view:
# Example: FIXED_CAMERA_POSITION = ([x1, y1, z1], [x2, y2, z2], [x3, y3, z3])
FIXED_CAMERA_POSITION = ([339045.7806324862, 76660.64997315472, -1061559.9153945546], 
                         [548556.3180566452, 254317.10972604417, 186306.30049153234], 
                         [0.03458397080983578, -0.9902187843317729, 0.13516918332049796])

# High-resolution sizing metrics
dpi = 2000  # High base DPI to keep spatial coordinates and layout crisp
mm = 1 / 25.4
width_px = int(40 * mm * dpi)
height_px = int(40 * mm * dpi)

IMAGE_DPI_SCALE = 2


def visualize_multiple_neurons(file_paths, colors=None, opacity=1.0, out_name='Neuron.png', camera_pos=None):
    """
    Plots SWC neuron files in PyVista using either an interactive preview or a fixed camera position.
    """
    color_list = colors

    print(f"Processing {len(file_paths)} neurons...")

    neuron_data = []

    # 1. Vectorized loading loop
    for idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        try:
            columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
            data = pd.read_csv(file_path, sep=r'\s+', comment='#', names=columns)

            points = data[['x', 'y', 'z']].values
            radii = data['radius'].values

            skeleton = pv.PolyData(points)
            skeleton.point_data['radius'] = radii

            sphere_source = pv.Sphere(radius=1.0, phi_resolution=8, theta_resolution=8)
            nodes = skeleton.glyph(scale='radius', geom=sphere_source, factor=1.0, orient=False)

            neuron_color = color_list[idx % len(color_list)]
            neuron_data.append((nodes, neuron_color))

        except Exception as e:
            print(f"Skipping {file_path} due to an error: {e}")

    if not neuron_data:
        print("No valid neuron geometry loaded. Aborting.")
        return

    # 2. FIXED CAMERA MODE (Batch Off-Screen Rendering)
    if camera_pos is not None:

        print(camera_pos)
        output_img = os.path.join(OUTPUT_DIR, out_name)
        os.makedirs(os.path.dirname(output_img), exist_ok=True)

        print(f"Rendering off-screen with fixed camera orientation...")
        plotter = pv.Plotter(window_size=[width_px, height_px], off_screen=True)
        plotter.background_color = "white"

        for nodes, neuron_color in neuron_data:
            plotter.add_mesh(nodes, color=neuron_color, opacity=opacity, smooth_shading=True)

        plotter.camera_position = camera_pos
        plotter.reset_camera_clipping_range()
        plotter.enable_anti_aliasing('ssaa')
        plotter.render()

        plotter.screenshot(output_img, scale=IMAGE_DPI_SCALE)
        print(f"✔ Fixed-camera render saved to: {output_img}")
        plotter.close()
        return

    # 3. INTERACTIVE MODE (Initial alignment & coordinate gathering)
    plotter = pv.Plotter(window_size=[width_px, height_px])
    plotter.background_color = "white"

    for nodes, neuron_color in neuron_data:
        plotter.add_mesh(nodes, color=neuron_color, opacity=opacity, smooth_shading=True)

    print("\nOpening Interactive 3D Canvas.")
    print(" -> Frame your neuron layout. When satisfied, press 'q' to reveal saving configurations.")
    
    plotter.enable_anti_aliasing('ssaa')
    plotter.show()

    try:
        raw_cam_pos = plotter.camera_position
        final_camera_pos = (list(raw_cam_pos[0]), list(raw_cam_pos[1]), list(raw_cam_pos[2]))
    except Exception:
        final_camera_pos = None

    save_prompt = input("\nWould you like to save an ultra high-quality picture from this perspective? (yes/no): ").strip().lower()

    if save_prompt in ['yes', 'y'] and final_camera_pos is not None:
        output_img = os.path.join(OUTPUT_DIR, out_name)
        os.makedirs(os.path.dirname(output_img), exist_ok=True)

        screenshot_plotter = pv.Plotter(window_size=[width_px, height_px], off_screen=True)
        screenshot_plotter.background_color = "white"
        screenshot_plotter.camera_position = final_camera_pos

        for nodes, neuron_color in neuron_data:
            screenshot_plotter.add_mesh(nodes, color=neuron_color, opacity=opacity, smooth_shading=True)

        screenshot_plotter.reset_camera_clipping_range()
        screenshot_plotter.enable_anti_aliasing('ssaa')
        screenshot_plotter.render()

        screenshot_plotter.screenshot(output_img, scale=IMAGE_DPI_SCALE)
        print(f"✔ Render saved to: {output_img}")
        
        # Log parameters
        log_filepath = os.path.join(OUTPUT_DIR, "neuron_render_log.csv")
        file_is_new = not os.path.exists(log_filepath)
        with open(log_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            if file_is_new:
                writer.writerow(["Saved_Image", "Camera_Position", "Camera_Focal_Point", "Camera_View_Up"])
            writer.writerow([os.path.basename(output_img), final_camera_pos[0], final_camera_pos[1], final_camera_pos[2]])
        print(f"✔ Camera parameters successfully logged to: {log_filepath}")
        print(f"-> Set FIXED_CAMERA_POSITION = {final_camera_pos} to reuse this exact angle!")

        screenshot_plotter.close()

    plotter.close()


if __name__ == "__main__":

    Top_IDs = ['720575940629513222', '720575940626979621', '720575940628908548',
               '720575940628307026', '720575940636543668', '720575940622516852',
               '720575940626403917', '720575940628038808', '720575940631190016']

    cmap = LinearSegmentedColormap.from_list("customgradient1", ["#4CD2FF", "#3080ffff"])
    colors = [cmap(i / (len(Top_IDs) - 1)) for i in range(len(Top_IDs))]

    for i, neuronID in enumerate(Top_IDs):
        print(f'\n--- Processing neuron {i+1}/{len(Top_IDs)}: {neuronID} ---')

        my_neuron_paths = [f"{BASE_FILE_PATH}{neuronID}.swc"]

        visualize_multiple_neurons(
            my_neuron_paths,
            colors=[colors[i]],
            opacity=1.0,
            out_name=f'Neuron_{i+1}_{neuronID}.png',
            camera_pos=FIXED_CAMERA_POSITION
        )