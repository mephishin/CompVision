import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# Класс для обработки изображений
class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    # Метод открывает изображение из файла
    def open_image(self, filename):
        try:
            self.original_image = Image.open(filename)
            self.processed_image = self.original_image.copy()
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось открыть изображение.")
            return False

    # Метод применяет масштабирование к изображению
    def apply_scaling(self, scale_x, scale_y):
        try:
            self.processed_image = self.original_image.resize((int(scale_x), int(scale_y)), Image.BILINEAR)
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось применить масштабирование.")
            return False

    # Метод применяет проекцию к изображению
    def apply_projection(self, points):
        try:
            # Проекция фрагмента изображения на произвольную плоскость
            self.processed_image = self.original_image.transform(
                (300, 300), Image.QUAD, points, resample=Image.BILINEAR
            )
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось выполнить проекцию.")
            return False

    # Метод применяет сдвиг к изображению    
    def apply_translation(self, translate_x, translate_y):
        try:
            self.processed_image = self.original_image.transform(
                self.original_image.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
            )
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось применить сдвиг.")
            return False

    # Метод применяет отражение к изображению   
    def apply_flip(self, direction):
        try:
            if direction == "horizontal":
                self.processed_image = self.original_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif direction == "vertical":
                self.processed_image = self.original_image.transpose(Image.FLIP_TOP_BOTTOM)
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось применить отражение.")
            return False

    # Метод применяет поворот к изображению
    def apply_rotation(self, angle, center):
        try:
            self.processed_image = self.original_image.rotate(angle, resample=Image.BILINEAR, center=center)
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось применить поворот.")
            return False


# Класс представляет графический интерфейс для обработки изображений
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработчик изображений")

        self.image_processor = ImageProcessor()

        self.left_panel = tk.LabelFrame(self, text="Origin")
        self.left_panel.grid(row=0, column=0, padx=10, pady=10)

        self.right_panel = tk.LabelFrame(self, text="After changes")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10)

        self.load_button = tk.Button(self, text="Load image", command=self.load_images)
        self.load_button.grid(row=1, columnspan=2, pady=10)

        self.scale_x_label = tk.Label(self, text="X scale:")
        self.scale_x_label.grid(row=2, column=0, padx=5)
        self.scale_x_entry = tk.Entry(self, width=10)
        self.scale_x_entry.grid(row=2, column=1, padx=5)
        self.scale_y_label = tk.Label(self, text="Y scale:")
        self.scale_y_label.grid(row=3, column=0, padx=5)
        self.scale_y_entry = tk.Entry(self, width=10)
        self.scale_y_entry.grid(row=3, column=1, padx=5)

        self.scale_button = tk.Button(self, text="Scale", command=self.scale_image)
        self.scale_button.grid(row=4, columnspan=2, pady=10)

        self.angle_label = tk.Label(self, text="Rotation angle (grad):")
        self.angle_label.grid(row=5, column=0, padx=5)
        self.angle_entry = tk.Entry(self, width=10)
        self.angle_entry.grid(row=5, column=1, padx=5)

        self.rotate_button = tk.Button(self, text="Rotate", command=self.rotate_image)
        self.rotate_button.grid(row=6, columnspan=2, pady=10)

        self.flip_horizontal_button = tk.Button(self, text="Flip horizontally",
                                                command=lambda: self.flip_image("horizontal"))
        self.flip_horizontal_button.grid(row=7, columnspan=2, pady=5)

        self.flip_vertical_button = tk.Button(self, text="Flip vertical",
                                              command=lambda: self.flip_image("vertical"))
        self.flip_vertical_button.grid(row=8, columnspan=2, pady=5)

        self.left_panel.grid_propagate(False)
        self.right_panel.grid_propagate(False)

    # Метод загружает изображение из файла
    def load_images(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            success = self.image_processor.open_image(file_path)
            if success:
                self.show_images()

    # Метод выполняет масштабирование изображения
    def scale_image(self):
        scale_x = self.scale_x_entry.get()
        scale_y = self.scale_y_entry.get()
        if scale_x and scale_y:
            success = self.image_processor.apply_scaling(float(scale_x), float(scale_y))
            if success:
                self.show_images()

    # Метод выполняет поворот изображения
    def rotate_image(self):
        angle = self.angle_entry.get()
        if angle:
            angle = float(angle)
            center = (self.image_processor.original_image.width // 2, self.image_processor.original_image.height // 2)
            success = self.image_processor.apply_rotation(angle, center)
            if success:
                self.show_images()

    # Метод применяет отражение к изображению в указанном направлении
    def flip_image(self, direction):
        success = self.image_processor.apply_flip(direction)
        if success:
            self.show_images()

    # Метод обновляет отображение изображений на графическом интерфейсе
    def show_images(self):
        original_image_tk = original_image_tk = ImageTk.PhotoImage(self.image_processor.original_image)
        processed_image_tk = ImageTk.PhotoImage(self.image_processor.processed_image)

        self.original_label = tk.Label(self.left_panel, image=original_image_tk)
        self.original_label.image = original_image_tk
        self.original_label.grid(row=0, column=0)

        self.processed_label = tk.Label(self.right_panel, image=processed_image_tk)
        self.processed_label.image = processed_image_tk
        self.processed_label.grid(row=0, column=0)

        self.left_panel.config(width=original_image_tk.width(), height=original_image_tk.height())
        self.right_panel.config(width=processed_image_tk.width(), height=processed_image_tk.height())

    # Метод применяет поворот к исходному изображению на заданный угол вокруг указанного центра.
    def apply_rotation(self, angle, center):
        try:
            self.processed_image = self.original_image.rotate(angle, resample=Image.BILINEAR, center=center)
            return True
        except:
            messagebox.showerror("Ошибка", "Не удалось применить поворот.")
            return False

    # Метод выполняет проекцию фрагмента исходного изображения на произвольную плоскость, определяемую четырьмя точками.
    def apply_projection(self, points):
        try:
            width, height = self.original_image.size
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            dst_points = np.float32([tuple(coord) for coord in points])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            processed_image_np = cv2.warpPerspective(np.array(self.original_image), matrix, (width, height))
            self.processed_image = Image.fromarray(processed_image_np)
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось выполнить проекцию: " + str(e))
            return False

    # Метод rotate_image вызывает метод apply_rotation для применения поворота к изображению с углом, указанным
    # пользователем через интерфейс приложения.
    def rotate_image(self):
        angle = self.angle_entry.get()
        if angle:
            angle = float(angle)
            center = (self.image_processor.original_image.width // 2, self.image_processor.original_image.height // 2)
            success = self.image_processor.apply_rotation(angle, center)
            if success:
                self.show_images()

    # Метод вызывает метод apply_projection для применения проекции к изображению с использованием координат,
    # указанных пользователем через интерфейс приложения.
    def project_image(self):
        points_str = self.points_entry.get()
        if points_str:
            # Преобразование строки с координатами в список кортежей
            points_list = [tuple(map(int, point.split(','))) for point in points_str.split()]
            if len(points_list) == 4:
                success = self.image_processor.apply_projection(points_list)
                if success:
                    self.show_images()
            else:
                messagebox.showerror("Ошибка", "Неверное количество точек. Необходимо указать 4 точки.")


# Запуск приложения
if __name__ == "__main__":
    app = App()
    app.mainloop()
