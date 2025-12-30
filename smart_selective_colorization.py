import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import cv2

# ==================== SIMPLE SEMANTIC SEGMENTATION MODEL ====================

class SimpleSegmentationModel(nn.Module):
    """A simple UNet-like architecture for semantic segmentation"""
    def __init__(self, num_classes=3):  # 0: background, 1: foreground, 2: other
        super(SimpleSegmentationModel, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 16 from enc1 + 16 from dec2
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )
        
    def forward(self, x):
        # Encoder path
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        
        # Decoder path
        dec2_out = self.dec2(enc2_out)
        # Skip connection
        dec2_out = torch.cat([dec2_out, enc1_out], dim=1)
        output = self.dec1(dec2_out)
        
        return output

# ==================== TARGETED COLORIZATION ====================

class TargetedColorizer:
    def __init__(self):
        self.model = SimpleSegmentationModel(num_classes=3)
        # Load pre-trained weights or initialize
        self.model.eval()
        
        # Color maps for different regions
        self.color_maps = {
            'background': [(0, 0, 255), (255, 0, 0)],  # Blue to Red gradient
            'foreground': [(0, 255, 0), (255, 255, 0)],  # Green to Yellow gradient
            'all': [(100, 100, 100), (255, 255, 255)]  # Gray to White
        }
        
    def segment_image(self, image):
        """Perform semantic segmentation on grayscale image"""
        # Convert PIL to tensor
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            segmentation = torch.argmax(output, dim=1).squeeze().numpy()
        
        # Resize segmentation back to original size
        segmentation = cv2.resize(segmentation, image.size, interpolation=cv2.INTER_NEAREST)
        
        return segmentation
    
    def colorize_region(self, grayscale_img, segmentation, region_type, color_intensity=0.5):
        """Colorize specific region based on segmentation mask"""
        # Convert to numpy array
        img_array = np.asarray(grayscale_img.convert('RGB'))
        
        # Create color overlay
        overlay = np.zeros_like(img_array, dtype=np.float32)
        
        # Determine mask based on region type
        if region_type == 'background':
            mask = (segmentation == 0)
        elif region_type == 'foreground':
            mask = (segmentation == 1)
        elif region_type == 'all':
            mask = np.ones(segmentation.shape, dtype=bool)
        else:  # custom region from GUI brush
            mask = segmentation
        
        if mask.sum() > 0:  # If there are pixels to colorize
            # Create gradient color
            rows, cols = np.where(mask)
            if len(rows) > 0:
                # Simple vertical gradient for demonstration
                norm_rows = (rows - rows.min()) / (rows.max() - rows.min() + 1e-7)
                
                for i, (r, c) in enumerate(zip(rows, cols)):
                    t = norm_rows[i] if len(rows) > 1 else 0.5
                    color = self.get_gradient_color(t, region_type)
                    overlay[r, c] = color
        
        # Blend with original
        colorized = img_array * (1 - color_intensity) + overlay * color_intensity
        colorized = np.clip(colorized, 0, 255).astype(np.uint8)
        
        return Image.fromarray(colorized)
    
    def get_gradient_color(self, t, region_type):
        """Get color from gradient based on position"""
        if region_type not in self.color_maps:
            region_type = 'all'
        
        colors = self.color_maps[region_type]
        if len(colors) == 1:
            return colors[0]
        
        # Interpolate between colors
        idx = t * (len(colors) - 1)
        idx1 = int(idx)
        idx2 = min(idx1 + 1, len(colors) - 1)
        
        frac = idx - idx1
        color = [
            int(colors[idx1][i] * (1 - frac) + colors[idx2][i] * frac)
            for i in range(3)
        ]
        
        return color

# ==================== GUI APPLICATION ====================

class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Targeted Colorization")
        self.root.geometry("900x600")
        
        self.colorizer = TargetedColorizer()
        self.current_image = None
        self.segmentation = None
        self.brush_mask = None
        self.brush_size = 20
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Buttons
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Segment", command=self.segment_image).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Clear Brush", command=self.clear_brush).grid(row=0, column=2, padx=5)
        
        # Region selection
        ttk.Label(control_frame, text="Colorize Region:").grid(row=0, column=3, padx=10)
        self.region_var = tk.StringVar(value="foreground")
        region_combo = ttk.Combobox(control_frame, textvariable=self.region_var, 
                                   values=["foreground", "background", "all", "brush"])
        region_combo.grid(row=0, column=4, padx=5)
        
        # Color intensity slider
        ttk.Label(control_frame, text="Intensity:").grid(row=0, column=5, padx=10)
        self.intensity_var = tk.DoubleVar(value=0.5)
        intensity_slider = ttk.Scale(control_frame, from_=0, to=1, variable=self.intensity_var,
                                    command=self.update_colorization)
        intensity_slider.grid(row=0, column=6, padx=5)
        
        # Brush size slider
        ttk.Label(control_frame, text="Brush Size:").grid(row=0, column=7, padx=10)
        self.brush_var = tk.IntVar(value=20)
        brush_slider = ttk.Scale(control_frame, from_=5, to=100, variable=self.brush_var,
                                command=lambda v: setattr(self, 'brush_size', int(float(v))))
        brush_slider.grid(row=0, column=8, padx=5)
        
        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Canvas for image display
        self.canvas = tk.Canvas(content_frame, width=800, height=500, bg='gray')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Load an image to begin")
        self.status_label.grid(row=2, column=0, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
    
    def load_image(self):
        """Load an image file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if filepath:
            self.current_image = Image.open(filepath)
            # Resize if too large
            if self.current_image.size[0] > 800 or self.current_image.size[1] > 500:
                self.current_image.thumbnail((800, 500))
            
            self.segmentation = None
            self.brush_mask = None
            self.display_image(self.current_image)
            self.status_label.config(text=f"Loaded: {filepath.split('/')[-1]}")
    
    def segment_image(self):
        """Perform segmentation on current image"""
        if self.current_image:
            self.status_label.config(text="Segmenting...")
            self.root.update()
            
            # Convert to grayscale for segmentation
            grayscale = self.current_image.convert('L')
            self.segmentation = self.colorizer.segment_image(grayscale)
            
            self.status_label.config(text="Segmentation complete! Select region to colorize.")
            self.update_colorization()
    
    def update_colorization(self, *args):
        """Update the colorized preview"""
        if self.current_image is None or self.segmentation is None:
            return
        
        region_type = self.region_var.get()
        intensity = self.intensity_var.get()
        
        # Get mask for selected region
        if region_type == 'brush' and self.brush_mask is not None:
            mask = self.brush_mask
        else:
            mask = self.segmentation
        
        # Colorize
        colorized = self.colorizer.colorize_region(
            self.current_image,
            mask,
            region_type,
            intensity
        )
        
        self.display_image(colorized)
    
    def display_image(self, image):
        """Display image on canvas"""
        self.display_img = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            400, 250,
            image=self.display_img,
            anchor=tk.CENTER
        )
    
    def paint(self, event):
        """Handle brush painting on canvas"""
        if self.current_image is None or self.region_var.get() != 'brush':
            return
        # Initialize brush mask if needed
        if self.brush_mask is None:
            if self.segmentation is not None:
                mask_shape = self.segmentation.shape
            else:
                mask_shape = (self.current_image.height, self.current_image.width)

            self.brush_mask = np.zeros(mask_shape, dtype=bool)
        
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_x = int(event.x * self.current_image.width / canvas_width)
        img_y = int(event.y * self.current_image.height / canvas_height)
        
        # Update brush mask
        size = self.brush_size
        y_start = max(0, img_y - size)
        y_end = min(self.brush_mask.shape[0], img_y + size)
        x_start = max(0, img_x - size)
        x_end = min(self.brush_mask.shape[1], img_x + size)
        
        # Create circular brush
        yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
        dist = ((yy - img_y) ** 2 + (xx - img_x) ** 2) ** 0.5
        brush_circle = dist <= size
        
        self.brush_mask[y_start:y_end, x_start:x_end] |= brush_circle
        
        self.update_colorization()
    
    def clear_brush(self):
        """Clear the brush mask"""
        self.brush_mask = None
        if self.segmentation is not None:
            self.update_colorization()

# ==================== MAIN EXECUTION ====================

def main():
    # Create a simple dummy model file or use a real one
    print("Initializing Targeted Colorization Application...")
    print("Note: This is a simplified implementation.")
    print("For production, you would need to:")
    print("1. Train the segmentation model on a proper dataset")
    print("2. Use a more sophisticated colorization approach")
    print("3. Add proper model loading/saving")
    
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()