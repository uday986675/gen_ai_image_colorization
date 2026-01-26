import cv2
import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class RealTimeVideoColorizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model = None
        self.models = {}
        self.is_running = False
        self.cap = None
        self.video_path = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.processed_queue = queue.Queue(maxsize=2)
        self.fps_history = deque(maxlen=30)
        
        # Initialize models
        self._initialize_models()
        
        # Create GUI
        self._create_gui()
    
    def _initialize_models(self):
        """Initialize different colorization models"""
        # Model 1: Fast CNN-based model (Balanced speed/quality)
        self.models['FastCNN'] = FastCNNColorizer().to(self.device).eval()
        
        # Model 2: Lightweight U-Net (Fastest)
        self.models['LightUNet'] = LightUNetColorizer().to(self.device).eval()
        
        # Model 3: Enhanced model (Best quality, slower)
        self.models['Enhanced'] = EnhancedColorizer().to(self.device).eval()
        
        self.current_model = self.models['LightUNet']
        
        # Load pre-trained weights if available
        self._load_model_weights()
    
    def _load_model_weights(self):
        """Load pre-trained weights for models"""
        # Note: The saved weights are for a different model architecture (conditional colorization)
        # These video models use random weights for now
        print("Video colorization models initialized with random weights (no pre-trained weights available)")
    
    def _create_gui(self):
        """Create and run the GUI"""
        self.root = tk.Tk()
        self.gui = VideoColorizerGUI(self.root, self)
    
    def capture_frames(self):
        """Capture frames from video source"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Put frame in queue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            # Control frame rate
            time.sleep(0.001)
    
    def process_frames(self):
        """Process frames for colorization"""
        processor = VideoProcessor(self.current_model, self.device)
        
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Colorize
                start_time = time.time()
                colorized = processor.process_frame(gray_frame)
                end_time = time.time()
                
                # Calculate FPS
                processing_time = end_time - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                
                # Put processed frame in queue
                try:
                    self.processed_queue.put_nowait(
                        (colorized, frame, avg_fps)
                    )
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

class FastCNNColorizer(nn.Module):
    """Fast CNN-based colorization model"""
    def __init__(self):
        super(FastCNNColorizer, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Middle processing
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()  # Output ab channels in Lab color space
        )
    
    def forward(self, gray):
        features = self.encoder(gray)
        features = self.middle(features)
        ab = self.decoder(features)
        return ab

class LightUNetColorizer(nn.Module):
    """Lightweight U-Net for real-time colorization"""
    def __init__(self):
        super(LightUNetColorizer, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Downsample
        self.down1 = conv_block(1, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        
        # Upsample with skip connections
        self.up1 = conv_block(128 + 64, 64)
        self.up2 = conv_block(64 + 32, 32)
        
        # Final layer
        self.final = nn.Conv2d(32, 2, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Downsample path
        conv1 = self.down1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)
        
        conv3 = self.down3(pool2)
        
        # Upsample path with skip connections
        up1 = self.upsample(conv3)
        merge1 = torch.cat([up1, conv2], dim=1)
        conv4 = self.up1(merge1)
        
        up2 = self.upsample(conv4)
        merge2 = torch.cat([up2, conv1], dim=1)
        conv5 = self.up2(merge2)
        
        output = self.final(conv5)
        return torch.tanh(output)

class EnhancedColorizer(nn.Module):
    """Enhanced model with better quality (slower)"""
    def __init__(self):
        super(EnhancedColorizer, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Downsample
        self.down1 = conv_block(1, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        
        # Upsample with skip connections
        self.up3 = conv_block(512 + 256, 256)
        self.up2 = conv_block(256 + 128, 128)
        self.up1 = conv_block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, 2, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Downsample path
        conv1 = self.down1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)
        
        conv3 = self.down3(pool2)
        pool3 = self.pool(conv3)
        
        conv4 = self.down4(pool3)
        
        # Upsample path with skip connections
        up3 = self.upsample(conv4)
        merge3 = torch.cat([up3, conv3], dim=1)
        conv5 = self.up3(merge3)
        
        up2 = self.upsample(conv5)
        merge2 = torch.cat([up2, conv2], dim=1)
        conv6 = self.up2(merge2)
        
        up1 = self.upsample(conv6)
        merge1 = torch.cat([up1, conv1], dim=1)
        conv7 = self.up1(merge1)
        
        output = self.final(conv7)
        return torch.tanh(output)

class VideoProcessor:
    def __init__(self, colorizer, device):
        self.colorizer = colorizer
        self.device = device
        
    def process_frame(self, gray_frame):
        """Process single frame for colorization"""
        # Convert to tensor
        gray_tensor = self._preprocess_frame(gray_frame)
        
        with torch.no_grad():
            # Get ab channels
            ab_tensor = self.colorizer(gray_tensor)
            
            # Convert back to RGB
            colorized = self._lab_to_rgb(gray_tensor, ab_tensor)
        
        return colorized
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Resize for model
        h, w = gray.shape
        new_h, new_w = 256, 256  # Fixed size for faster processing
        gray_resized = cv2.resize(gray, (new_w, new_h))
        
        # Normalize
        gray_normalized = gray_resized / 255.0
        gray_tensor = torch.FloatTensor(gray_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return gray_tensor
    
    def _lab_to_rgb(self, L, ab):
        """Convert Lab to RGB"""
        # Combine L and ab
        ab_upsampled = torch.nn.functional.interpolate(ab, size=L.shape[2:], mode='bilinear')
        Lab = torch.cat([L, ab_upsampled], dim=1)
        
        # Convert to numpy and then to RGB
        Lab_np = Lab.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        Lab_np[:, :, 0] = Lab_np[:, :, 0] * 100
        Lab_np[:, :, 1:] = Lab_np[:, :, 1:] * 127
        
        # Convert to RGB
        rgb = cv2.cvtColor(Lab_np.astype(np.float32), cv2.COLOR_Lab2RGB)
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb
    
class VideoColorizerGUI:
    def __init__(self, root, colorizer_app):
        self.root = root
        self.app = colorizer_app
        self.setup_styles()
        self.setup_gui()
        
    def setup_styles(self):
        """Set up modern styling similar to Amazon's design"""
        self.style = ttk.Style()
        
        # Configure overall theme
        self.style.theme_use('default')
        
        # Amazon-like color palette
        self.amazon_blue = '#007185'  # Amazon's primary blue
        self.amazon_orange = '#ff9900'  # Amazon's accent orange
        self.amazon_light_gray = '#f3f3f3'  # Light background
        self.amazon_dark_gray = '#666666'  # Text gray
        
        # Configure button styles
        self.style.configure('Amazon.TButton',
                           background=self.amazon_blue,
                           foreground='white',
                           font=('Arial', 10, 'bold'),
                           padding=(10, 5),
                           relief='flat',
                           borderwidth=0)
        
        self.style.map('Amazon.TButton',
                      background=[('active', '#005a70'), ('pressed', '#004d5f')])
        
        # Configure label styles
        self.style.configure('Amazon.TLabel',
                           background='white',
                           foreground='black',
                           font=('Arial', 10))
        
        self.style.configure('Header.TLabel',
                           background=self.amazon_blue,
                           foreground='white',
                           font=('Arial', 16, 'bold'),
                           padding=(10, 5))
        
        # Configure frame styles
        self.style.configure('Card.TFrame',
                           background='white',
                           relief='solid',
                           borderwidth=1)
        
        self.style.configure('ControlPanel.TFrame',
                           background=self.amazon_light_gray,
                           relief='solid',
                           borderwidth=1)
        
        # Configure combobox
        self.style.configure('Amazon.TCombobox',
                           font=('Arial', 10),
                           fieldbackground='white')
        
        # Root window styling
        self.root.configure(bg='white')
        
    def setup_gui(self):
        self.root.title("AI Video Colorizer")
        self.root.geometry("1400x1000")  # Increased height
        self.root.configure(bg='white')
        
        # Header
        header_frame = ttk.Frame(self.root, style='Header.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_label = ttk.Label(header_frame, text="AI Video Colorizer", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Main content frame
        main_frame = ttk.Frame(self.root, style='Card.TFrame')
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
        # Control Panel
        control_frame = ttk.Frame(main_frame, style='ControlPanel.TFrame', padding="20")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        # Model Selection Section
        model_section = ttk.Frame(control_frame, style='Card.TFrame', padding="15")
        model_section.pack(fill='x', pady=(0, 10))
        
        ttk.Label(model_section, text="Colorization Model", 
                 font=('Arial', 12, 'bold'), background='white').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.model_var = tk.StringVar(value="LightUNet")
        model_combo = ttk.Combobox(model_section, textvariable=self.model_var, 
                                  values=list(self.app.models.keys()), 
                                  state="readonly", style='Amazon.TCombobox', font=('Arial', 10))
        model_combo.grid(row=1, column=0, padx=(0, 10))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Source Selection Section
        source_section = ttk.Frame(control_frame, style='Card.TFrame', padding="15")
        source_section.pack(fill='x', pady=(0, 10))
        
        ttk.Label(source_section, text="Video Source", 
                 font=('Arial', 12, 'bold'), background='white').grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        webcam_btn = ttk.Button(source_section, text="📹 Start Webcam", 
                               command=lambda: self.start_video_source('webcam'), style='Amazon.TButton')
        webcam_btn.grid(row=1, column=0, padx=(0, 10))
        
        load_btn = ttk.Button(source_section, text="🎬 Load Video File", 
                             command=self.load_video_file, style='Amazon.TButton')
        load_btn.grid(row=1, column=1)
        
        # Status Section
        status_section = ttk.Frame(control_frame, style='Card.TFrame', padding="15")
        status_section.pack(fill='x')
        
        ttk.Label(status_section, text="Status", 
                 font=('Arial', 12, 'bold'), background='white').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.fps_label = ttk.Label(status_section, text="FPS: 0.0", 
                                  font=('Arial', 11), background='white', foreground=self.amazon_dark_gray)
        self.fps_label.grid(row=1, column=0, sticky="w", padx=(0, 20))
        
        self.start_button = ttk.Button(status_section, text="▶️ Start Processing", 
                                      command=self.toggle_video, style='Amazon.TButton')
        self.start_button.grid(row=1, column=1, padx=(10, 0))
        
        screenshot_btn = ttk.Button(status_section, text="📸 Screenshot", 
                                   command=self.save_screenshot, style='Amazon.TButton')
        screenshot_btn.grid(row=1, column=2, padx=(10, 0))
        
        # Video Display Area with Scrollbar
        display_container = ttk.Frame(main_frame, style='Card.TFrame')
        display_container.grid(row=1, column=0, sticky="nsew", pady=(20, 0))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(display_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for scrolling
        self.display_canvas = tk.Canvas(display_container, bg='white', yscrollcommand=scrollbar.set)
        self.display_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.display_canvas.yview)
        
        # Frame inside canvas
        display_frame = ttk.Frame(self.display_canvas, style='Card.TFrame', padding="20")
        self.display_canvas.create_window((0, 0), window=display_frame, anchor="nw")
        
        ttk.Label(display_frame, text="Video Preview", 
                 font=('Arial', 14, 'bold'), background='white').grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Original frame
        original_frame = ttk.Frame(display_frame, style='Card.TFrame', padding="10")
        original_frame.grid(row=1, column=0, padx=(0, 10), sticky="n")
        ttk.Label(original_frame, text="Original (Grayscale)", 
                 font=('Arial', 11, 'bold'), background='white').pack()
        self.original_label = ttk.Label(original_frame, background='black')
        self.original_label.pack(pady=(5, 0))
        
        # Colorized frame
        colorized_frame = ttk.Frame(display_frame, style='Card.TFrame', padding="10")
        colorized_frame.grid(row=1, column=1, padx=(10, 0), sticky="n")
        ttk.Label(colorized_frame, text="Colorized Output", 
                 font=('Arial', 11, 'bold'), background='white').pack()
        self.colorized_label = ttk.Label(colorized_frame, background='black')
        self.colorized_label.pack(pady=(5, 0))
        
        # Configure canvas scrolling
        display_frame.bind("<Configure>", lambda e: self.display_canvas.configure(scrollregion=self.display_canvas.bbox("all")))
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
    def on_model_change(self, event):
        model_name = self.model_var.get()
        self.app.current_model = self.app.models[model_name]
        print(f"Switched to model: {model_name}")
    
    def start_video_source(self, source):
        if source == 'webcam':
            self.app.video_path = 0  # Webcam index
            self.toggle_video()
    
    def load_video_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.app.video_path = file_path
            self.toggle_video()
    
    def toggle_video(self):
        if not self.app.is_running:
            self.start_video()
        else:
            self.stop_video()
    
    def start_video(self):
        self.app.is_running = True
        self.start_button.config(text="Stop")
        
        # Start video capture
        self.app.cap = cv2.VideoCapture(self.app.video_path)
        
        # Start processing threads
        self.app.capture_thread = threading.Thread(target=self.app.capture_frames)
        self.app.process_thread = threading.Thread(target=self.app.process_frames)
        
        self.app.capture_thread.start()
        self.app.process_thread.start()
        
        # Start display update
        self.update_display()
    
    def stop_video(self):
        self.app.is_running = False
        self.start_button.config(text="Start")
        if self.app.cap:
            self.app.cap.release()
    
    def update_display(self):
        if self.app.is_running:
            try:
                # Get processed frame
                processed_data = self.app.processed_queue.get_nowait()
                colorized_frame, original_frame, fps = processed_data
                
                # Update FPS display
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                
                # Convert frames for display
                original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                colorized_rgb = cv2.cvtColor(colorized_frame, cv2.COLOR_RGB2BGR)
                colorized_rgb = cv2.cvtColor(colorized_rgb, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                display_size = (640, 480)
                original_display = cv2.resize(original_rgb, display_size)
                colorized_display = cv2.resize(colorized_rgb, display_size)
                
                # Convert to PhotoImage
                original_img = ImageTk.PhotoImage(
                    Image.fromarray(original_display))
                colorized_img = ImageTk.PhotoImage(
                    Image.fromarray(colorized_display))
                
                # Update labels
                self.original_label.config(image=original_img)
                self.original_label.image = original_img
                self.colorized_label.config(image=colorized_img)
                self.colorized_label.image = colorized_img
                
            except queue.Empty:
                pass
            
            # Schedule next update
            self.root.after(10, self.update_display)
    
    def save_screenshot(self):
        if hasattr(self, 'last_colorized_frame'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, 
                           cv2.cvtColor(self.last_colorized_frame, cv2.COLOR_RGB2BGR))
                print(f"Screenshot saved to {file_path}")

def main():
    # Initialize the colorizer
    print("Initializing Real-Time Video Colorizer...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create application
    app = RealTimeVideoColorizer()
    
    # Run application
    app.run()

if __name__ == "__main__":
    main()
