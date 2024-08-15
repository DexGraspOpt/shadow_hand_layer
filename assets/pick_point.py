import pyvista as pv

# Load the mesh
mesh = pv.read('hand.stl')

# Create a plotter and add the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='gray')

# Function to handle point selection
def pick_point():
    point_id = plotter.point_picker.point_id
    if point_id >= 0:
        selected_point = mesh.points[point_id]
        print(f"Selected point: {selected_point}")

        # Visualize the selected point
        sphere = pv.Sphere(radius=0.02, center=selected_point)
        sphere.paint_uniform_color((1, 0, 0))  # Red color
        plotter.add_mesh(sphere, render_points_as_spheres=True)
        plotter.render()

# Add a key event to enable picking
plotter.add_key_event('p', lambda: plotter.enable_point_picking(callback=pick_point))

# Display the mesh and start the interaction
plotter.show()