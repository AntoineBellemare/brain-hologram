import numpy as np
from vispy import app, scene
from vispy.scene import Node
from vispy.scene.visuals import Sphere, Volume
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy.color import Colormap
from scipy.ndimage import gaussian_filter
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt

# Load data
voxels_raw = np.load("voxels.npy").astype("float32")
ch_pos = np.load("ch_pos.npy").astype("float32")
voxels_raw = np.nan_to_num(voxels_raw, nan=0.0, posinf=0.0, neginf=0.0)
voxels_raw = np.abs(voxels_raw)

print(f"Loaded {len(voxels_raw)} frames")

# Create smooth colormap - deep blue -> purple -> magenta -> pink
colors = np.array([
    [0.02, 0.02, 0.12],     # Very dark blue
    [0.05, 0.05, 0.20],     # Dark blue
    [0.10, 0.08, 0.28],     # Deep blue
    [0.15, 0.10, 0.35],     # Blue-violet
    [0.22, 0.12, 0.42],     # Deep purple
    [0.30, 0.15, 0.48],     # Purple
    [0.38, 0.18, 0.52],     # Purple-magenta
    [0.48, 0.22, 0.55],     # Magenta-purple
    [0.58, 0.28, 0.58],     # Magenta
    [0.68, 0.35, 0.60],     # Light magenta
    [0.75, 0.42, 0.62],     # Magenta-pink
    [0.82, 0.50, 0.65],     # Pink-magenta
    [0.88, 0.60, 0.70],     # Soft pink
    [0.92, 0.70, 0.78],     # Light pink
])
COLORMAP = Colormap(colors)

# Default parameters
params = {
    'smoothing': 1.5,
    'gamma': 2.0,
    'step_size': 0.5,
    'clim_min': 0.0,
    'clim_max': 0.70,
    'contrast': 0.8,  # Lower for original normalization
    'rotation_speed': 2.0,
    'brightness': 1.0,  # Higher since original norm is already scaled well
    'percentile_min': 5.0,  # Less aggressive
    'percentile_max': 95.0,  # Less aggressive
    'frame_skip': 1,
    'log_scale': 0.0,  # Turn off - not needed with original norm
}

# Process voxels with current parameters
def process_voxels():
    voxels = voxels_raw.copy()
    
    # Logarithmic scaling to compress high values
    if params['log_scale'] > 0:
        voxels = voxels + 1e-8
        log_voxels = np.log(voxels + 1)
        voxels = (1 - params['log_scale']) * voxels + params['log_scale'] * log_voxels
    
    # Normalize per-frame with AGGRESSIVE clipping
    for i in range(len(voxels)):
        frame = voxels[i]
        vmin, vmax = np.percentile(frame, [params['percentile_min'], params['percentile_max']])
        if vmax > vmin:
            voxels[i] = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
    
    # Power law to push values DOWN
    voxels = voxels ** params['contrast']
    
    # Final brightness adjustment
    voxels = voxels * params['brightness']
    voxels = np.clip(voxels, 0, 1)
    
    # Smoothing AFTER processing for smooth gradients
    if params['smoothing'] > 0:
        for i in range(len(voxels)):
            voxels[i] = gaussian_filter(voxels[i], sigma=params['smoothing'])
    
    return voxels

voxels = process_voxels()

# Create canvas
canvas = scene.SceneCanvas(size=(1200, 800), show=True, bgcolor="black", keys='interactive')

# Create main layout widget
main_widget = QWidget()
main_layout = QHBoxLayout()
main_widget.setLayout(main_layout)

# Add vispy canvas to left side
main_layout.addWidget(canvas.native, stretch=3)

# Create control panel on right
control_panel = QWidget()
control_panel.setStyleSheet("""
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
""")
control_layout = QVBoxLayout()
control_layout.setSpacing(2)
control_layout.setContentsMargins(10, 10, 10, 10)
control_panel.setLayout(control_layout)
control_panel.setFixedWidth(380)  # Fixed width prevents resizing
main_layout.addWidget(control_panel)

# Title
title = QLabel("Hologram Controls")
title.setStyleSheet("""
    font-size: 18px; 
    font-weight: bold; 
    padding: 12px;
    background-color: #1a1a1a;
    color: #00ccff;
    border-radius: 4px;
    margin-bottom: 8px;
""")
title.setAlignment(Qt.AlignmentFlag.AlignCenter)
control_layout.addWidget(title)

# Store slider references
sliders = {}

def create_slider(name, label, min_val, max_val, default, scale=100):
    """Create a slider with label"""
    group = QGroupBox(label)
    group.setStyleSheet("""
        QGroupBox {
            font-size: 11px;
            font-weight: bold;
            padding: 8px;
            margin-top: 8px;
            border: 1px solid #444;
            border-radius: 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    group.setFixedHeight(70)  # Fixed height prevents resizing
    layout = QVBoxLayout()
    layout.setSpacing(4)
    layout.setContentsMargins(8, 15, 8, 8)
    
    value_label = QLabel(f"{default:.2f}")
    value_label.setStyleSheet("""
        font-weight: bold; 
        color: #00aaff; 
        font-size: 13px;
        padding: 2px;
    """)
    value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    value_label.setFixedHeight(20)
    
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(int(min_val * scale))
    slider.setMaximum(int(max_val * scale))
    slider.setValue(int(default * scale))
    slider.setTickPosition(QSlider.TickPosition.NoTicks)
    slider.setFixedHeight(24)
    slider.setStyleSheet("""
        QSlider::groove:horizontal {
            border: 1px solid #333;
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a2a, stop:1 #1a1a1a);
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0088cc, stop:1 #0066aa);
            border: 1px solid #005588;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00aaff, stop:1 #0088cc);
        }
    """)
    
    layout.addWidget(value_label)
    layout.addWidget(slider)
    group.setLayout(layout)
    
    sliders[name] = {'slider': slider, 'label': value_label, 'scale': scale}
    return group

# Add sliders
control_layout.addWidget(create_slider('smoothing', '1. Smoothing (σ)', 0, 5, params['smoothing']))
control_layout.addWidget(create_slider('gamma', '2. Gamma', 0.5, 10, params['gamma']))
control_layout.addWidget(create_slider('step_size', '3. Ray Step Size', 0.1, 2, params['step_size']))
control_layout.addWidget(create_slider('clim_min', '4. Color Min', 0, 0.5, params['clim_min']))
control_layout.addWidget(create_slider('clim_max', '5. Color Max', 0.2, 1, params['clim_max']))
control_layout.addWidget(create_slider('contrast', '6. Contrast (γ)', 0.3, 5, params['contrast']))
control_layout.addWidget(create_slider('rotation_speed', '7. Rotation Speed', 0, 10, params['rotation_speed']))
control_layout.addWidget(create_slider('brightness', '8. Brightness', 0.1, 2, params['brightness']))
control_layout.addWidget(create_slider('percentile_min', '9. Norm Min %', 0, 30, params['percentile_min']))
control_layout.addWidget(create_slider('percentile_max', '10. Norm Max %', 70, 100, params['percentile_max']))
control_layout.addWidget(create_slider('frame_skip', '11. Animation Speed', 1, 20, params['frame_skip'], scale=1))
control_layout.addWidget(create_slider('log_scale', '12. Log Scale Mix', 0, 1, params['log_scale']))

# Reset button
reset_btn = QPushButton("Reset Defaults")
reset_btn.setStyleSheet("""
    QPushButton {
        padding: 12px; 
        font-weight: bold; 
        font-size: 12px;
        background: #0066cc; 
        color: white;
        border: none;
        border-radius: 4px;
        margin-top: 10px;
    }
    QPushButton:hover {
        background: #0088ee;
    }
    QPushButton:pressed {
        background: #004488;
    }
""")
control_layout.addWidget(reset_btn)

# Reprocess button
reprocess_btn = QPushButton("Reprocess Voxels")
reprocess_btn.setStyleSheet("""
    QPushButton {
        padding: 12px; 
        font-weight: bold; 
        font-size: 12px;
        background: #cc6600; 
        color: white;
        border: none;
        border-radius: 4px;
        margin-top: 5px;
    }
    QPushButton:hover {
        background: #ee8800;
    }
    QPushButton:pressed {
        background: #884400;
    }
""")
control_layout.addWidget(reprocess_btn)

control_layout.addStretch()

# Setup vispy scene
view = canvas.central_widget.add_view()

first_frame = np.ascontiguousarray(voxels[0], dtype=np.float32)
holo = Volume(
    first_frame,
    parent=view.scene,
    method="additive",
    clim=(params['clim_min'], params['clim_max']),
    cmap=COLORMAP,
    interpolation='linear',
    gamma=params['gamma'],
    relative_step_size=params['step_size'],
)

holo.set_gl_state('additive', blend=True, depth_test=False, cull_face=False)
holo.transform = MatrixTransform()
center = np.array(first_frame.shape) / 2.0
holo.transform.translate(-center)

# Electrodes
elecs = Node()
for ch in ch_pos:
    sphere = Sphere(radius=0.2, method="ico", subdivisions=2, color="yellow")
    sphere.mesh.set_gl_state("translucent", depth_test=False)
    sphere.transform = STTransform(translate=ch)
    sphere.parent = elecs
elecs.transform = MatrixTransform()
elecs.transform.translate(-center)
view.add(elecs)

view.camera = "arcball"
view.camera.center = (0, 0, 0)

# Animation state
current_frame = [0]
rotation_angle = [0]

# Update handlers
def update_param(param_name):
    def handler(value):
        actual_value = value / sliders[param_name]['scale']
        params[param_name] = actual_value
        sliders[param_name]['label'].setText(f"{actual_value:.2f}")
        
        # Update volume properties that don't require reprocessing
        if param_name == 'gamma':
            holo.gamma = actual_value
        elif param_name == 'step_size':
            holo.relative_step_size = actual_value
        elif param_name in ['clim_min', 'clim_max']:
            holo.clim = (params['clim_min'], params['clim_max'])
    return handler

# Connect sliders
for name in sliders:
    sliders[name]['slider'].valueChanged.connect(update_param(name))

def reset_defaults():
    defaults = {
        'smoothing': 1.5, 'gamma': 2.0, 'step_size': 0.5,
        'clim_min': 0.0, 'clim_max': 0.70, 'contrast': 0.8,
        'rotation_speed': 2.0, 'brightness': 1.0,
        'percentile_min': 5.0, 'percentile_max': 95.0,
        'frame_skip': 1, 'log_scale': 0.0,
    }
    for name, value in defaults.items():
        sliders[name]['slider'].setValue(int(value * sliders[name]['scale']))

def reprocess():
    global voxels
    print("Reprocessing voxels with new parameters...")
    voxels = process_voxels()
    holo.set_data(voxels[current_frame[0]])
    print("Done!")

reset_btn.clicked.connect(reset_defaults)
reprocess_btn.clicked.connect(reprocess)

def update(ev):
    try:
        # Skip frames for faster animation
        skip = max(1, int(params['frame_skip']))
        i = (current_frame[0] + skip) % len(voxels)
        current_frame[0] = i
        
        frame_data = np.ascontiguousarray(voxels[i], dtype=np.float32)
        holo.set_data(frame_data)
        
        rotation_angle[0] += ev.dt * params['rotation_speed']
        holo.transform.reset()
        holo.transform.translate(-center)
        holo.transform.rotate(rotation_angle[0], (0, 0, 1))
        
        elecs.transform.reset()
        elecs.transform.translate(-center)
        elecs.transform.rotate(rotation_angle[0], (0, 0, 1))
    except Exception as e:
        print(f"Error: {e}")

timer = app.Timer(1 / 30, connect=update, start=True)

# Show the main widget
main_widget.show()
canvas.native.setParent(main_widget)

print("\n=== HOLOGRAM CONTROLS ===")
print("Adjust sliders to see real-time effects!")
print("Click 'Reprocess Voxels' after changing smoothing, contrast, brightness, or percentiles")

app.run()
