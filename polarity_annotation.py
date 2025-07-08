import os
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from vtk.util import numpy_support
from tkinter import filedialog, messagebox
import pandas as pd
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

# Function to read SWC files
def read_swc(file_path):
    data = np.loadtxt(file_path, comments='#')
    return data

# Convert SWC data to PyVista PolyData
def swc_to_polydata(swc_data):
    points = swc_data[:, 2:5] / 1000.0
    lines = []
    id_to_index = {int(row[0]): i for i, row in enumerate(swc_data)}
    radii = swc_data[:, 5] / 1000.0

    for row in swc_data:
        node_id = int(row[0])
        parent_id = int(row[6])
        if parent_id != -1:
            if parent_id in id_to_index and node_id in id_to_index:
                lines.extend([2, id_to_index[parent_id], id_to_index[node_id]])

    mesh = pv.PolyData(points, lines=lines)
    colors = np.zeros((len(points), 3))
    mesh.point_data["colors"] = colors

    return mesh

# Extract soma nodes and create spheres
def create_soma_spheres(swc_data):
    soma_nodes = swc_data[swc_data[:, 1] == 1]
    soma_points = soma_nodes[:, 2:5] / 1000.0
    soma_radii = soma_nodes[:, 5] / 1000.0

    soma_spheres = pv.PolyData()

    for i in range(len(soma_points)):
        sphere = pv.Sphere(center=soma_points[i], radius=soma_radii[i])
        soma_spheres += sphere

    return soma_spheres

# Pop up multiple input boxes for user to input annotations
def input_annotations(swc_file, annotation_types, default_values):
    dialog = QDialog()
    dialog.setWindowTitle(f"Input Annotations for File {swc_file}")
    layout = QVBoxLayout()

    inputs = {}
    for annotation_type in annotation_types:
        label = QLabel(f"Please enter {annotation_type} (default: {default_values.get(annotation_type, 'None')}):")
        layout.addWidget(label)
        input_field = QLineEdit()
        layout.addWidget(input_field)
        inputs[annotation_type] = input_field

    submit_button = QPushButton("Submit")
    layout.addWidget(submit_button)

    result = {}

    def on_submit():
        for annotation_type, input_field in inputs.items():
            value = input_field.text()
            result[annotation_type] = value if value else default_values.get(annotation_type, "")
        dialog.accept()

    submit_button.clicked.connect(on_submit)
    dialog.setLayout(layout)

    if dialog.exec_() == QDialog.Accepted:
        return result
    else:
        return None

# Calculate appropriate grid size
def calculate_grid_size(num_neurons):
    if num_neurons <= 0:
        return (1, 1)
    rows = int(np.ceil(np.sqrt(num_neurons)))
    cols = int(np.ceil(num_neurons / rows))
    return (rows, cols)

# Batch read SWC files and display
def display_neurons_in_grid(swc_folder, page, neurons_per_page, annotation_types=None, default_values=None, annotations=None):
    if annotation_types is None:
        annotation_types = []
    if default_values is None:
        default_values = {}
    if annotations is None:
        annotations = {}

    swc_files = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]
    total_neurons = len(swc_files)
    start_index = page * neurons_per_page
    end_index = min(start_index + neurons_per_page, total_neurons)
    swc_files = swc_files[start_index:end_index]

    num_neurons = len(swc_files)
    grid_size = calculate_grid_size(num_neurons)
    plotter = pv.Plotter(shape=grid_size)

    print(f"Total SWC files in folder: {total_neurons}")
    print(f"Number of SWC files displayed on current page: {num_neurons}")
    print(f"Calculated grid_size: {grid_size}")

    is_annotation_open = False

    def on_right_click(obj, event):
        nonlocal is_annotation_open
        if is_annotation_open:
            return

        x, y = obj.GetEventPosition()

        for i, renderer in enumerate(plotter.renderers):
            viewport = renderer.GetViewport()
            width, height = plotter.window_size

            x_min = int(viewport[0] * width)
            x_max = int(viewport[2] * width)
            y_min = int((1 - viewport[3]) * height)
            y_max = int((1 - viewport[1]) * height)

            if x_min <= x <= x_max and y_min <= y <= y_max:
                row = i // grid_size[1]
                col = i % grid_size[1]
                reversed_row = grid_size[0] - 1 - row
                file_index = reversed_row * grid_size[1] + col

                if file_index >= len(swc_files):
                    print("Clicked on empty area, no corresponding file")
                    return

                swc_file = swc_files[file_index]
                print(f"Right-clicked on file: {swc_file}")
                is_annotation_open = True

                annotation_results = input_annotations(swc_file, annotation_types, default_values)
                if annotation_results:
                    if swc_file not in annotations:
                        annotations[swc_file] = {}
                    for annotation_type in annotation_types:
                        annotations[swc_file][annotation_type] = annotation_results.get(annotation_type, default_values.get(annotation_type, ""))
                        print(f"Added annotation: {annotation_type} - {annotations[swc_file][annotation_type]}")

                is_annotation_open = False
                break

    #plotter.add_text("Right-click to input annotations", position=(0.5, 0.95), font_size=12, color="black")

    for i in range(grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        plotter.subplot(row, col)

        if i < len(swc_files):
            swc_file = swc_files[i]
            try:
                swc_data = read_swc(os.path.join(swc_folder, swc_file))
                neuron_mesh = swc_to_polydata(swc_data)
                soma_spheres = create_soma_spheres(swc_data)

                soma_center = np.mean(soma_spheres.points, axis=0)
                translation = -soma_center
                neuron_mesh.points += translation
                soma_spheres.points += translation

                plotter.add_mesh(neuron_mesh, color="black", line_width=2)
                plotter.add_mesh(soma_spheres, color="red", opacity=0.5)
                plotter.enable_trackball_style()

                plotter.add_text(swc_file, position="upper_left", font_size=10, color="red")

            except Exception as e:
                print(f"Error processing file {swc_file}: {e}")
        else:
            plotter.add_text("Empty", position=(0.5, 0.5), font_size=12, color="black")

    plotter.iren.add_observer('RightButtonPressEvent', on_right_click)
    plotter.show()

# Display a single neuron file
def display_single_neuron(swc_file):
    plotter = pv.Plotter()
    try:
        swc_data = read_swc(swc_file)
        neuron_mesh = swc_to_polydata(swc_data)
        soma_spheres = create_soma_spheres(swc_data)

        plotter.add_mesh(neuron_mesh, color="black", line_width=2)
        plotter.add_mesh(soma_spheres, color="red", opacity=0.5)
        plotter.show_axes()
        plotter.enable_trackball_style()
        plotter.add_text("Single Neuron Display", position=(0.5, 0.95), font_size=12, color="black")

        plotter.show()
    except Exception as e:
        print(f"Error processing file {swc_file}: {e}")

# GUI interface
class SWCViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SWC Viewer")

        # Annotation Type Settings Frame
        self.annotation_frame = tk.LabelFrame(root, text="Annotation Type Settings", padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.annotation_frame.pack(fill="x", padx=10, pady=5)

        self.annotation_types = []
        self.default_values = {}

        # Add Annotation Type Button
        self.add_annotation_button = tk.Button(self.annotation_frame, text="Add Annotation Type", command=self.add_annotation_type)
        self.add_annotation_button.pack(pady=5)

        # Separator between Annotation Type Settings and Single Neuron Display
        self.separator1 = ttk.Separator(root, orient=tk.HORIZONTAL)
        self.separator1.pack(fill="x", padx=10, pady=5)

        # Single Neuron Display Frame
        self.single_neuron_frame = tk.LabelFrame(root, text="Single Neuron Display", padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.single_neuron_frame.pack(fill="x", padx=10, pady=5)

        # Select and Display Single Neuron File Button
        self.single_file_button = tk.Button(self.single_neuron_frame, text="Select and Display Single Neuron File", command=self.display_single_neuron)
        self.single_file_button.pack(pady=5)

        # Separator between Single Neuron Display and Multi-Neuron Display
        self.separator2 = ttk.Separator(root, orient=tk.HORIZONTAL)
        self.separator2.pack(fill="x", padx=10, pady=5)

        # Multi-Neuron Display Frame
        self.multi_neuron_frame = tk.LabelFrame(root, text="Multi-Neuron Display", padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.multi_neuron_frame.pack(fill="x", padx=10, pady=5)

        # Select SWC Folder Button
        self.select_button = tk.Button(self.multi_neuron_frame, text="Select SWC Folder", command=self.select_folder)
        self.select_button.grid(row=0, column=0, columnspan=2, pady=5)

        # Neurons per Page Label and Entry
        self.neurons_per_page_label = tk.Label(self.multi_neuron_frame, text="Number of Neurons per Page:")
        self.neurons_per_page_label.grid(row=1, column=0, pady=5)

        self.neurons_per_page_entry = tk.Entry(self.multi_neuron_frame, width=10)
        self.neurons_per_page_entry.insert(0, "25")
        self.neurons_per_page_entry.grid(row=1, column=1, pady=5)

        # Current Page Label
        self.page_label = tk.Label(self.multi_neuron_frame, text="Current Page: 0")
        self.page_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Previous and Next Page Buttons
        self.prev_button = tk.Button(self.multi_neuron_frame, text="Previous Page", command=self.prev_page)
        self.prev_button.grid(row=3, column=0, pady=5)

        self.next_button = tk.Button(self.multi_neuron_frame, text="Next Page", command=self.next_page)
        self.next_button.grid(row=3, column=1, pady=5)

        # Display Neurons Button
        self.display_button = tk.Button(self.multi_neuron_frame, text="Display Neurons", command=self.display_neurons)
        self.display_button.grid(row=4, column=0, columnspan=2, pady=5)

        # Save Annotations Button
        self.save_button = tk.Button(self.multi_neuron_frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.grid(row=5, column=0, columnspan=2, pady=5)

        self.swc_folder = None
        self.current_page = 0
        self.neurons_per_page = 25
        self.annotations = {}  # Global storage for all annotations

    def select_folder(self):
        self.swc_folder = filedialog.askdirectory(title="Select SWC Folder")
        if self.swc_folder:
            print(f"Selected folder: {self.swc_folder}")

    def add_annotation_type(self):
        annotation_frame = tk.Frame(self.annotation_frame)
        annotation_frame.pack(pady=5)

        annotation_type_var = tk.StringVar()
        annotation_type_menu = tk.OptionMenu(annotation_frame, annotation_type_var, "annotation1", "annotation2", "annotation3", "annotation4", "annotation5")
        annotation_type_menu.pack(side=tk.LEFT, padx=5)

        default_value_entry = tk.Entry(annotation_frame, width=10)
        default_value_entry.pack(side=tk.LEFT, padx=5)

        def remove_annotation_type():
            annotation_frame.destroy()
            self.annotation_types.remove(annotation_type_var.get())
            del self.default_values[annotation_type_var.get()]

        remove_button = tk.Button(annotation_frame, text="Delete", command=remove_annotation_type)
        remove_button.pack(side=tk.LEFT, padx=5)

        def update_annotation_type(*args):
            if annotation_type_var.get() in self.annotation_types:
                self.default_values[annotation_type_var.get()] = default_value_entry.get()
            else:
                self.annotation_types.append(annotation_type_var.get())
                self.default_values[annotation_type_var.get()] = default_value_entry.get()

        annotation_type_var.trace_add("write", update_annotation_type)
        default_value_entry.bind("<KeyRelease>", lambda event: update_annotation_type())

    def display_neurons(self):
        if self.swc_folder:
            try:
                self.neurons_per_page = int(self.neurons_per_page_entry.get())
                if self.neurons_per_page <= 0:
                    raise ValueError("Number of neurons per page must be greater than 0")
            except ValueError as e:
                print(f"Invalid input for number of neurons per page: {e}")
                return

            display_neurons_in_grid(self.swc_folder, self.current_page, self.neurons_per_page, annotation_types=self.annotation_types, default_values=self.default_values, annotations=self.annotations)
        else:
            print("Please select an SWC folder first")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.page_label.config(text=f"Current Page: {self.current_page}")
            self.display_neurons()

    def next_page(self):
        if self.swc_folder:
            total_neurons = len([f for f in os.listdir(self.swc_folder) if f.endswith('.swc')])
            max_page = (total_neurons - 1) // self.neurons_per_page
            if self.current_page < max_page:
                self.current_page += 1
                self.page_label.config(text=f"Current Page: {self.current_page}")
                self.display_neurons()

    def save_annotations(self):
        if not self.swc_folder:
            messagebox.showwarning("Warning", "Please select an SWC folder first")
            return

        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return

        data = []
        swc_files = [f for f in os.listdir(self.swc_folder) if f.endswith('.swc')]
        for swc_file in swc_files:
            row = {"Name": swc_file}
            if swc_file in self.annotations:
                for annotation_type in self.annotation_types:
                    row[annotation_type] = self.annotations[swc_file].get(annotation_type, self.default_values.get(annotation_type, ""))
            else:
                for annotation_type in self.annotation_types:
                    row[annotation_type] = self.default_values.get(annotation_type, "")
            data.append(row)

        df = pd.DataFrame(data)
        output_path = os.path.join(self.swc_folder, "annotations.csv")
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Annotations saved to: {output_path}")
        messagebox.showinfo("Save Successful", f"Annotations saved to: {output_path}")

    def display_single_neuron(self):
        swc_file = filedialog.askopenfilename(title="Select Single SWC File", filetypes=[("SWC Files", "*.swc")])
        if swc_file:
            print(f"Selected file: {swc_file}")
            display_single_neuron(swc_file)

# Main program
if __name__ == "__main__":
    app = QApplication([])

    root = tk.Tk()
    app_gui = SWCViewerApp(root)
    root.mainloop()