import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Mesh, Markers
from vispy.visuals.transforms import MatrixTransform
from skimage import measure
import matplotlib.pyplot as plt

ROTATION_SPEED = 2
SHOW_ELECTRODES = True
NUM_LAYERS = 5  # Multiple semi-transparent layers for hologram effect
COLORMAP = plt.cm.viridis  # Less yellow, more blue-green-purple

voxels = np.load("voxels.npy").astype("float32")
ch_pos = np.load("ch_pos.npy").astype("float32")

# Clean and normalize
voxels = np.nan_to_num(voxels, nan=0.0, posinf=0.0, neginf=0.0)
voxels = np.abs(voxels) ** 2
vmin, vmax = np.percentile(voxels, [5, 99.5])
voxels = np.clip(voxels, vmin, vmax)
voxels = (voxels - vmin) / (vmax - vmin + 1e-8)
voxels = voxels ** 0.5

print(f"Voxel shape: {voxels.shape}")

canvas = scene.SceneCanvas(size=(800, 600), show=True, bgcolor="black", keys='interactive')
view = canvas.central_widget.add_view()

# Create multiple isosurface layers for hologram effect
mesh_layers = []
iso_levels = np.linspace(0.3, 0.8, NUM_LAYERS)

try:
    for idx, level in enumerate(iso_levels):
        verts, faces, normals, values = measure.marching_cubes(voxels[0], level=level)
        
        # Color with gradient + transparency
        intensity = (idx + 1) / NUM_LAYERS
        colors = COLORMAP(verts[:, 2] / voxels.shape[3])
        colors[:, 3] = 0.15 + 0.1 * intensity  # Semi-transparent
        
        mesh = Mesh(vertices=verts, faces=faces, vertex_colors=colors, shading='smooth')
        mesh.set_gl_state('translucent', depth_test=True, cull_face=False, blend=True, 
                          blend_func=('src_alpha', 'one'))  # Additive blending
        mesh.transform = MatrixTransform()
        
        center = np.array(voxels[0].shape) / 2
        mesh.transform.translate(-center)
        view.add(mesh)
        mesh_layers.append(mesh)
    
    print(f"Created {len(mesh_layers)} hologram layers")
except Exception as e:
    print(f"Error creating layers: {e}")

# Add glowing particles at high-intensity voxels
def create_particles(volume, threshold=0.8, max_points=3000):
    """Extract high-intensity voxels as glowing particles"""
    high_intensity = np.where(volume > threshold)
    
    if len(high_intensity[0]) == 0:
        return None, None
    
    # Subsample if too many points
    indices = np.random.choice(len(high_intensity[0]), 
                              min(len(high_intensity[0]), max_points), 
                              replace=False)
    
    positions = np.column_stack([high_intensity[0][indices], 
                                 high_intensity[1][indices], 
                                 high_intensity[2][indices]]).astype(np.float32)
    
    intensities = volume[high_intensity[0][indices], 
                         high_intensity[1][indices], 
                         high_intensity[2][indices]]
    
    # Color and size based on intensity
    colors = COLORMAP(intensities)
    colors[:, 3] = intensities * 0.5  # More transparent
    sizes = 2 + intensities * 6  # Smaller particles
    
    return positions, colors, sizes

# Create initial particles
pos, cols, szs = create_particles(voxels[0])
if pos is not None:
    particles = Markers()
    particles.set_data(pos, face_color=cols, size=szs, edge_width=0)
    particles.set_gl_state('translucent', depth_test=False, blend=True,
                          blend_func=('src_alpha', 'one'))
    particles.transform = MatrixTransform()
    center = np.array(voxels[0].shape) / 2
    particles.transform.translate(-center)
    view.add(particles)
else:
    particles = None

# Add electrode markers (subtle)
if SHOW_ELECTRODES:
    center_pos = np.array([voxels.shape[1], voxels.shape[2], voxels.shape[3]]) / 2
    elec_markers = Markers()
    elec_markers.set_data(ch_pos - center_pos, face_color=(0.3, 0.8, 1.0, 0.4),  # Cyan, semi-transparent
                         size=4, edge_width=0)
    elec_markers.transform = MatrixTransform()
    view.add(elec_markers)

view.camera = "arcball"
view.camera.center = (0, 0, 0)
view.camera.distance = 80

current_frame = [0]
rotation = [0]

def update(ev):
    try:
        i = (current_frame[0] + 1) % len(voxels)
        current_frame[0] = i
        
        # Update each isosurface layer
        for idx, (mesh, level) in enumerate(zip(mesh_layers, iso_levels)):
            try:
                verts, faces, normals, values = measure.marching_cubes(voxels[i], level=level)
                
                intensity = (idx + 1) / NUM_LAYERS
                colors = COLORMAP(verts[:, 2] / voxels.shape[3])
                colors[:, 3] = 0.15 + 0.1 * intensity
                
                mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors)
            except:
                pass  # Skip if no surface at this level
        
        # Update particles
        if particles is not None:
            pos, cols, szs = create_particles(voxels[i])
            if pos is not None:
                particles.set_data(pos - center, face_color=cols, size=szs)
        
        # Rotate everything
        rotation[0] += ev.dt * ROTATION_SPEED
        center = np.array(voxels[0].shape) / 2
        
        for mesh in mesh_layers:
            mesh.transform.reset()
            mesh.transform.translate(-center)
            mesh.transform.rotate(rotation[0], (0, 0, 1))
        
        if particles is not None:
            particles.transform.reset()
            particles.transform.translate(-center)
            particles.transform.rotate(rotation[0], (0, 0, 1))
        
        if SHOW_ELECTRODES:
            elec_markers.transform.reset()
            elec_markers.transform.rotate(rotation[0], (0, 0, 1))
        
    except Exception as e:
        pass  # Silent fail for smoother animation

timer = app.Timer(1 / 30, connect=update, start=True)

print("\nHologram visualization ready!")
print("Controls: Drag to rotate, Scroll to zoom, Q to quit")

app.run()
