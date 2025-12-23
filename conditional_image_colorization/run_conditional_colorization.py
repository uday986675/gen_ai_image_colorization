#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, Canvas
from PIL import Image, ImageTk, ImageDraw
import numpy as np


class SimpleColorizationNet:  # placeholder to match notebook structure
    def __init__(self):
        pass


class ConditionalColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conditional Image Colorization")
        self.root.geometry("1000x700")

        # Model placeholder
        self.model = SimpleColorizationNet()

        self.original_image = None
        self.grayscale_image = None
        self.condition_mask = None
        self.color_conditions = {}
        self.current_color = "#FF0000"
        self.drawing = False
        self.current_mask_points = []
        self.current_mask_id = 0

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        image_frame = ttk.Frame(self.root, padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Choose Color", command=self.choose_color).pack(side=tk.LEFT, padx=5)

        self.color_display = Canvas(control_frame, width=30, height=30, bg=self.current_color)
        self.color_display.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_size = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=5, to=50, width=10, textvariable=self.brush_size).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start Drawing", command=self.start_drawing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Colorize", command=self.colorize_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Result", command=self.save_image).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Conditions:").pack(side=tk.LEFT, padx=5)
        self.condition_listbox = tk.Listbox(control_frame, height=1, width=20)
        self.condition_listbox.pack(side=tk.LEFT, padx=5)

        self.canvas_original = Canvas(image_frame, width=400, height=400, bg="gray")
        self.canvas_original.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(image_frame, text="Original / Mask").grid(row=1, column=0)

        self.canvas_result = Canvas(image_frame, width=400, height=400, bg="gray")
        self.canvas_result.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(image_frame, text="Colorized Result").grid(row=1, column=1)

        self.canvas_original.bind("<Button-1>", self.start_draw_event)
        self.canvas_original.bind("<B1-Motion>", self.draw_event)
        self.canvas_original.bind("<ButtonRelease-1>", self.stop_draw_event)

    def load_image(self):
        # Ensure root is updated so dialogs attach correctly
        self.root.update()

        file_path = filedialog.askopenfilename(
            parent=self.root,
            initialdir='.',
            title='Select an image',
            filetypes=[("Image files", ("*.jpg", "*.jpeg", "*.png", "*.bmp")), ("All files", "*.*")]
        )

        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.grayscale_image = self.original_image.convert("L")
            self.condition_mask = Image.new("RGB", self.original_image.size, (128, 128, 128))
            self.color_conditions.clear()
            self.current_mask_id = 0
            self.condition_listbox.delete(0, tk.END)
            self.display_images()

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose color for condition")[1]
        if color:
            self.current_color = color
            self.color_display.config(bg=color)

    def start_drawing(self):
        self.drawing = True
        self.current_mask_points = []

    def start_draw_event(self, event):
        if self.drawing and self.original_image:
            self.current_mask_points = [(event.x, event.y)]
            self.draw_on_mask(event.x, event.y)

    def draw_event(self, event):
        if self.drawing and self.original_image and self.current_mask_points:
            self.current_mask_points.append((event.x, event.y))
            self.draw_on_mask(event.x, event.y)

    def stop_draw_event(self, event):
        if self.drawing and self.current_mask_points and self.original_image:
            self.current_mask_id += 1
            self.color_conditions[self.current_mask_id] = (self.current_color, self.current_mask_points.copy())
            self.condition_listbox.insert(tk.END, f"Condition {self.current_mask_id}: {self.current_color}")
            self.current_mask_points = []

    def draw_on_mask(self, x, y):
        if not self.original_image:
            return

        color_rgb = self.hex_to_rgb(self.current_color)
        draw = ImageDraw.Draw(self.condition_mask)

        img_width, img_height = self.original_image.size
        canvas_width = max(1, self.canvas_original.winfo_width())
        canvas_height = max(1, self.canvas_original.winfo_height())

        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height

        img_x = int(x * scale_x)
        img_y = int(y * scale_y)
        brush_size = int(self.brush_size.get() * scale_x)

        draw.ellipse([img_x - brush_size, img_y - brush_size, img_x + brush_size, img_y + brush_size], fill=color_rgb)
        self.display_images()

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def display_images(self):
        if self.original_image:
            display_img = self.original_image.copy()
            mask_display = self.condition_mask.copy().resize(self.original_image.size)
            blended = Image.blend(display_img.convert("RGB"), mask_display.convert("RGB"), 0.3)

            display_size = (400, 400)
            blended_display = blended.resize(display_size, Image.LANCZOS)
            self.photo_original = ImageTk.PhotoImage(blended_display)
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.photo_original)

            grayscale_display = self.grayscale_image.resize(display_size, Image.LANCZOS)
            self.photo_result = ImageTk.PhotoImage(grayscale_display)
            self.canvas_result.create_image(0, 0, anchor=tk.NW, image=self.photo_result)

    def clear_all(self):
        if self.original_image:
            self.condition_mask = Image.new("RGB", self.original_image.size, (128, 128, 128))
            self.color_conditions.clear()
            self.current_mask_id = 0
            self.condition_listbox.delete(0, tk.END)
            self.display_images()

    def colorize_image(self):
        if not self.original_image or not self.color_conditions:
            return

        img_array = np.array(self.grayscale_image)
        condition_array = np.array(self.condition_mask)

        height, width = img_array.shape
        result = np.zeros((height, width, 3), dtype=np.uint8)
        result[:, :, 0] = img_array
        result[:, :, 1] = img_array
        result[:, :, 2] = img_array

        mask_indices = np.any(condition_array != [128, 128, 128], axis=-1)
        if np.any(mask_indices):
            alpha = 0.7
            result[mask_indices] = (alpha * condition_array[mask_indices] + (1 - alpha) * result[mask_indices]).astype(np.uint8)

        result_img = Image.fromarray(result)
        result_display = result_img.resize((400, 400), Image.LANCZOS)
        self.photo_result = ImageTk.PhotoImage(result_display)
        self.canvas_result.create_image(0, 0, anchor=tk.NW, image=self.photo_result)
        self.result_image = result_img

    def save_image(self):
        if hasattr(self, 'result_image'):
            file_path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                self.result_image.save(file_path)


def main():
    root = tk.Tk()
    app = ConditionalColorizationApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
