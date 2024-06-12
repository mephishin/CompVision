import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


# Метод для загрузки изображения
def load_image(file_path):
    try:
        image = cv2.imread(file_path)
        return image
    except Exception as e:
        print("Ошибка загрузки изображения:", e)
        return None


# Метод для отображения отдельного цветового канала
def display_channel(image, channel):
    channel_index = {"Красный": 2, "Зеленый": 1, "Синий": 0}.get(channel, 0)
    channel_image = image[:, :, channel_index]
    return channel_image


# Метод для преобразования изображения в оттенки серого
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# Метод для добавления эффекта сепии к изображению
def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return sepia_image


# Метод для изменения яркости и контраста изображения
def brightness_contrast(image, brightness=0, contrast=0):
    alpha = (contrast + 100) / 100.0
    beta = brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


# Метод для выполнения логических операций над двумя изображениями
def logical_operations(image1, image2, operation):
    if operation == "И":
        result_image = cv2.bitwise_and(image1, image2)
    elif operation == "ИЛИ":
        result_image = cv2.bitwise_or(image1, image2)
    elif operation == "ИСКЛЮЧАЮЩЕЕ ИЛИ":
        result_image = cv2.bitwise_xor(image1, image2)
    elif operation == "НЕ":
        result_image = cv2.bitwise_not(image1)
    return result_image


# Метод для преобразования изображения в цветовое пространство HSV и изменения его компонент
def hsv_transformation(image, hue=0, saturation=0, value=0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + saturation, 0, 255)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + value, 0, 255)
    transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return transformed_image


# Метод для применения медианного размытия к изображению
def median_blur(image, kernel_size=3):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


# Метод для применения оконного фильтра к изображению
def window_filter(image, kernel):
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


# Метод для создания эффекта акварели на основе двух изображений
def watercolor(image1, image2, brightness=0, contrast=0, blend=0.5):
    adjusted_image = brightness_contrast(image1, brightness, contrast)
    blended_image = cv2.addWeighted(adjusted_image, blend, image2, 1 - blend, 0)
    return blended_image


# Метод для создания мультфильм-эффекта
def cartoon(image, threshold=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray_image, 5)
    edges = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    cartoon_image = cv2.bitwise_and(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), image)
    return cartoon_image


# Метод для получения списка доступных операций
def get_operation_list():
    return [
        "Отображение канала",
        "Оттенки серого",
        "Сепия",
        "Яркость и контраст",
        "Логические операции",
        "Преобразование HSV",
        "Медианное размытие",
        "Оконный фильтр",
        "Акварель",
        "Мультфильм"
    ]


# Метод для отображения изображения на холсте
def display_image(image, canvas):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo


class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Приложение для обработки изображений")

        self.canvas1 = tk.Canvas(self, width=400, height=400)
        self.canvas1.grid(row=0, column=0)

        self.canvas2 = tk.Canvas(self, width=400, height=400)
        self.canvas2.grid(row=0, column=1)

        self.load_button1 = tk.Button(self, text="Load image 1", command=lambda: self.load_image(1))
        self.load_button1.grid(row=1, column=0)

        self.load_button2 = tk.Button(self, text="Load image 2", command=lambda: self.load_image(2))
        self.load_button2.grid(row=1, column=1)

        self.selected_operation = tk.StringVar()
        self.operation_menu = tk.OptionMenu(self, self.selected_operation, *get_operation_list())
        self.operation_menu.grid(row=2, column=0)

        self.apply_button = tk.Button(self, text="Apply", command=self.apply_operation)
        self.apply_button.grid(row=2, column=1)

        self.image1 = None
        self.image2 = None
        self.selected_operation.set(get_operation_list()[0])

    # Метод для загрузки изображения из файла
    def load_image(self, image_num):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = load_image(file_path)
            if image_num == 1:
                self.image1 = image
                display_image(self.image1, self.canvas1)
            else:
                self.image2 = image
                display_image(self.image2, self.canvas2)

    # Метод для применения выбранной операции к изображению
    def apply_operation(self):
        operation = self.selected_operation.get()
        if operation == "Отображение канала":
            channel = "Красный"
            processed_image = display_channel(self.image1, channel)
        elif operation == "Оттенки серого":
            processed_image = grayscale(self.image1)
        elif operation == "Сепия":
            processed_image = sepia(self.image1)
        elif operation == "Яркость и контраст":
            processed_image = brightness_contrast(self.image1, brightness=10, contrast=10)
        elif operation == "Логические операции":
            processed_image = logical_operations(self.image1, self.image2, "И")
        elif operation == "Преобразование HSV":
            processed_image = hsv_transformation(self.image1, hue=20, saturation=50, value=50)
        elif operation == "Медианное размытие":
            processed_image = median_blur(self.image1, kernel_size=5)
        elif operation == "Оконный фильтр":
            kernel = np.ones((3, 3), dtype=np.float32) / 9
            processed_image = window_filter(self.image1, kernel)
        elif operation == "Акварель":
            processed_image = watercolor(self.image1, self.image2, brightness=10, contrast=10, blend=0.5)
        elif operation == "Мультфильм":
            processed_image = cartoon(self.image1, threshold=10)
        if processed_image is not None:
            display_image(processed_image, self.canvas2)
        else:
            print("Ошибка обработки изображения.")


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
