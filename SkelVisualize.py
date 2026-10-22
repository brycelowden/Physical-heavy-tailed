import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pyvista as pv

BASE_FILE_PATH = '[...]' # Folder containing FAFB SWC neuron skeletons
OUTPUT_DIR = '[...]' # Screenshot output

FIXED_CAMERA_POSITION = ([339045.7806324862, 76660.64997315472, -1061559.9153945546], 
                         [548556.3180566452, 254317.10972604417, 186306.30049153234], 
                         [0.03458397080983578, -0.9902187843317729, 0.13516918332049796])

dpi = 2000 
mm = 1 / 25.4
width_px = int(40 * mm * dpi)
height_px = int(40 * mm * dpi)

IMAGE_DPI_SCALE = 2


def visualize_multiple_neurons(file_paths, colors=None, opacity=1.0, out_name='Neuron.png', camera_pos=None):
    color_list = colors
    neuron_data = []

    for idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print("File not found")
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
            print(f"File error: {e}")

    if not neuron_data:
        print("Error constructing skeleton")
        return

    if camera_pos is not None:

        print(camera_pos)
        output_img = os.path.join(OUTPUT_DIR, out_name)
        os.makedirs(os.path.dirname(output_img), exist_ok=True)

        plotter = pv.Plotter(window_size=[width_px, height_px], off_screen=True)
        plotter.background_color = "white"

        for nodes, neuron_color in neuron_data:
            plotter.add_mesh(nodes, color=neuron_color, opacity=opacity, smooth_shading=True)

        plotter.camera_position = camera_pos
        plotter.reset_camera_clipping_range()
        plotter.enable_anti_aliasing('ssaa')
        plotter.render()

        plotter.screenshot(output_img, scale=IMAGE_DPI_SCALE)
        print("Render saved")
        plotter.close()
        return

    plotter = pv.Plotter(window_size=[width_px, height_px])
    plotter.background_color = "white"

    for nodes, neuron_color in neuron_data:
        plotter.add_mesh(nodes, color=neuron_color, opacity=opacity, smooth_shading=True)

    print("\nOpening Interactive 3D Canvas.")
    
    plotter.enable_anti_aliasing('ssaa')
    plotter.show()

    try:
        raw_cam_pos = plotter.camera_position
        final_camera_pos = (list(raw_cam_pos[0]), list(raw_cam_pos[1]), list(raw_cam_pos[2]))
    except Exception:
        final_camera_pos = None

    save_prompt = input("\nSave render? (yes/no): ").strip().lower()

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
        print("Render saved")

        screenshot_plotter.close()

    plotter.close()

if __name__ == "__main__":

    # Use any neuron IDs here. These are the top 9 neurons by connection count used in figure 1.

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
