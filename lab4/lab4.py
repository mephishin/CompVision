import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def display_image(image, canvas):
    canvas.delete("all")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.photo = photo
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Программа обработки изображений")

        self.original_image = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        # Основной фрейм
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # Фрейм для кнопок и настроек
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.LEFT)

        # Кнопка загрузки изображения
        self.load_button = tk.Button(self.controls_frame, text="Load image", command=self.load_image)
        self.load_button.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка предварительной обработки
        self.preprocess_button = tk.Button(self.controls_frame, text="Pre handle", command=self.preprocess_image, state=tk.DISABLED)
        self.preprocess_button.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка поиска контуров
        self.contours_button = tk.Button(self.controls_frame, text="Find contours", command=self.find_contours, state=tk.DISABLED)
        self.contours_button.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка поиска примитивов
        self.primitives_button = tk.Button(self.controls_frame, text="Find primitives", command=self.find_primitives, state=tk.DISABLED)
        self.primitives_button.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка сохранения результата
        self.save_button = tk.Button(self.controls_frame, text="Save result", command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, padx=5, pady=5)

        # Поле для настройки порогового значения
        self.threshold_label = tk.Label(self.controls_frame, text="Threshold:")
        self.threshold_label.pack(padx=5, pady=5)
        self.threshold_scale = tk.Scale(self.controls_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.threshold_scale.set(127)
        self.threshold_scale.pack(fill=tk.X, padx=5, pady=5)

        # Поле для настройки минимальной площади контура
        self.min_area_label = tk.Label(self.controls_frame, text="Contours min square:")
        self.min_area_label.pack(padx=5, pady=5)
        self.min_area_entry = tk.Entry(self.controls_frame)
        self.min_area_entry.insert(0, "100")
        self.min_area_entry.pack(fill=tk.X, padx=5, pady=5)

        # Поле для отображения числа примитивов
        self.primitive_count_label = tk.Label(self.controls_frame, text="Quantity of pimitives:")
        self.primitive_count_label.pack(padx=5, pady=5)
        self.primitive_count_value = tk.Label(self.controls_frame, text="0")
        self.primitive_count_value.pack(padx=5, pady=5)

        # Поля для отображения изображений
        self.original_canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.original_canvas.pack(side=tk.LEFT, padx=10)

        self.processed_canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.processed_canvas.pack(side=tk.RIGHT, padx=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            display_image(self.original_image, self.original_canvas)
            display_image(self.processed_image, self.processed_canvas)
            self.preprocess_button.config(state=tk.NORMAL)
            self.contours_button.config(state=tk.NORMAL)
            self.primitives_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)

    def preprocess_image(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            self.processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
            display_image(self.processed_image, self.processed_canvas)

    def find_contours(self):
        if self.processed_image is not None:
            gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, self.threshold_scale.get(), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            try:
                min_area = int(self.min_area_entry.get())
            except ValueError:
                min_area = 100

            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            image_with_contours = self.original_image.copy()
            cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)
            display_image(image_with_contours, self.processed_canvas)

    def find_primitives(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                min_area = int(self.min_area_entry.get())
            except ValueError:
                min_area = 100

            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            triangle_count = 0
            rectangle_count = 0
            circle_count = 0

            for contour in filtered_contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                num_sides = len(approx)

                if num_sides == 3:
                    shape = "Треугольник"
                    triangle_count += 1
                elif num_sides == 4:
                    shape = "Четырёхугольник"
                    rectangle_count += 1
                else:
                    shape = "Окружность"
                    circle_count += 1

                cv2.drawContours(self.processed_image, [contour], 0, (0, 255, 0), 2)
                cv2.putText(self.processed_image, shape, (approx.ravel()[0], approx.ravel()[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            display_image(self.processed_image, self.processed_canvas)
            self.primitive_count_value.config(text=f"Треугольников: {triangle_count}, Четырёхугольников: {rectangle_count}, Окружностей: {circle_count}")

    def save_result(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def update_buttons_state(self):
        state = tk.NORMAL if self.original_image is not None else tk.DISABLED
        self.preprocess_button.config(state=state)
        self.contours_button.config(state=state)
        self.primitives_button.config(state=state)
        self.save_button.config(state=state)


root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
