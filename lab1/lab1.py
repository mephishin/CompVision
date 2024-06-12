import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Приложение для обработки изображений")

        # Создаем элементы интерфейса
        self.canvas1 = tk.Canvas(self, width=400, height=400)
        self.canvas1.grid(row=0, column=0)

        self.canvas2 = tk.Canvas(self, width=400, height=400)
        self.canvas2.grid(row=0, column=1)

        self.load_button = tk.Button(self, text="Load iamge", command=self.load_image)
        self.load_button.grid(row=1, column=0)

        self.canny_label = tk.Label(self, text="Canny filter threshold:")
        self.canny_label.grid(row=1, column=1)
        self.canny_threshold_var = tk.DoubleVar()
        self.canny_threshold_entry = tk.Entry(self, textvariable=self.canny_threshold_var)
        self.canny_threshold_entry.grid(row=1, column=2)

        self.threshold_label = tk.Label(self, text="Threshold:")
        self.threshold_label.grid(row=2, column=1)
        self.threshold_var = tk.DoubleVar()
        self.threshold_entry = tk.Entry(self, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=2, column=2)

        self.process_button = tk.Button(self, text="Handle image", command=self.process_image)
        self.process_button.grid(row=3, column=0)

        self.capture_button = tk.Button(self, text="Capture video", command=self.capture_video)
        self.capture_button.grid(row=3, column=1)

        self.play_button = tk.Button(self, text="Play video", command=self.play_video)
        self.play_button.grid(row=3, column=2)

        self.current_image = None
        self.video_capture = None

    # Метод для загрузки изображения
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image()

    # Метод для обработки изображения
    def process_image(self):
        if self.current_image is not None:
            canny_threshold = self.canny_threshold_var.get()
            threshold = self.threshold_var.get()

            # Применяем фильтр Кэнни и пороговый фильтр
            canny_image = cv2.Canny(self.current_image, canny_threshold, canny_threshold * 2)
            _, threshold_image = cv2.threshold(self.current_image, threshold, 255, cv2.THRESH_BINARY)

            self.display_processed_images(canny_image, threshold_image)

    # Метод для захвата видеопотока
    def capture_video(self):
        self.video_capture = cv2.VideoCapture(0)
        self.play_video()

    # Метод для воспроизведения видео
    def play_video(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                canny_image = cv2.Canny(frame, self.canny_threshold_var.get(), self.canny_threshold_var.get() * 2)
                _, threshold_image = cv2.threshold(frame, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
                self.display_processed_images(canny_image, threshold_image)
                self.after(10, self.play_video)

    # Метод для отображения изображения
    def display_image(self):
        if self.current_image is not None:
            # Преобразуем изображение из OpenCV в PIL
            pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            # Создаем объект Tkinter.PhotoImage из PIL-изображения
            photo = ImageTk.PhotoImage(pil_image)
            self.canvas1.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas1.image = photo

    # Метод для отображения обработанных изображений
    def display_processed_images(self, canny_image, threshold_image):
        # Преобразуем изображения из OpenCV в PIL
        pil_canny_image = Image.fromarray(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB))
        pil_threshold_image = Image.fromarray(cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB))

        # Создаем объекты Tkinter.PhotoImage из PIL-изображений
        canny_photo = ImageTk.PhotoImage(pil_canny_image)
        threshold_photo = ImageTk.PhotoImage(pil_threshold_image)

        self.canvas1.create_image(0, 0, anchor=tk.NW, image=canny_photo)
        self.canvas1.image = canny_photo

        self.canvas2.create_image(0, 0, anchor=tk.NW, image=threshold_photo)
        self.canvas2.image = threshold_photo

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
