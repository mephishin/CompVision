import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class FeatureDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Приложение для обнаружения особенностей")
        
        self.video_source = 0  # По умолчанию - веб-камера
        self.capture = None
        self.frame = None
        
        # Создание виджетов
        self.original_label = tk.Label(master)
        self.original_label.grid(row=0, column=0)
        
        self.processed_label = tk.Label(master)
        self.processed_label.grid(row=0, column=1)
        
        self.load_button = tk.Button(master, text="Загрузить изображение", command=self.load_image)
        self.load_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.find_features_button = tk.Button(master, text="Найти особенности", command=self.find_features)
        self.find_features_button.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.track_features_button = tk.Button(master, text="Отследить особенности", command=self.track_features)
        self.track_features_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.match_features_button = tk.Button(master, text="Сопоставить особенности", command=self.match_features)
        self.match_features_button.grid(row=4, column=0, columnspan=2, pady=5)
        
    # Метод для загрузки изображения
    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.original_image = cv2.imread(self.file_path)
            self.display_image(self.original_image, self.original_label)
        
    # Метод для отображения изображения в виджете
    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk
    
    # Метод для поиска особенностей
    def find_features(self):
        if self.original_image is None:
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        self.keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in self.keypoints]
        self.keypoints_image = cv2.drawKeypoints(self.original_image, self.keypoints, None, color=(0, 255, 0))
        self.display_image(self.keypoints_image, self.processed_label)
        
    # Метод для отслеживания особенностей
    def track_features(self):
        if self.original_image is None:
            return
        if self.keypoints is None:
            self.find_features()
        
        old_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        new_image_path = filedialog.askopenfilename()
        if new_image_path:
            new_image = cv2.imread(new_image_path)
            new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            p0 = np.array([point.pt for point in self.keypoints], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None)

            # Добавляем отладочный вывод для проверки размерности st
            print("Размерность st:", st.shape)

            good_indices = np.where(st==1)[0]  # Находим индексы хороших точек
            good_new = p1[good_indices]  # Выбираем хорошие новые точки
            good_old = p0[good_indices]  # Выбираем соответствующие старые точки
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.circle(new_image, (int(a), int(b)), 5, (0, 255, 0), -1)
                cv2.line(new_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            
            self.display_image(new_image, self.processed_label)
    
    # Метод для сопоставления особенностей
    def match_features(self):
        if self.original_image is None:
            return
        new_image_path = filedialog.askopenfilename()
        if new_image_path:
            new_image = cv2.imread(new_image_path)
            gray1 = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)
            matching_result = cv2.drawMatches(self.original_image, kp1, new_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_image(matching_result, self.processed_label)

def main():
    root = tk.Tk()
    app = FeatureDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
