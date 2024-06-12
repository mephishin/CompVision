import sys
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\song_\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image



class ImageProcessingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Приложение обработки изображений")
        self.setGeometry(100, 100, 800, 600)

        self.left_label = QtWidgets.QLabel()
        self.right_label = QtWidgets.QLabel()

        self.load_image_button = QtWidgets.QPushButton("Загрузить изображение")
        self.detect_text_button = QtWidgets.QPushButton("Выделить текст")
        self.display_text_button = QtWidgets.QPushButton("Отобразить текст")
        self.detect_faces_button = QtWidgets.QPushButton("Обнаружить лица")
        self.apply_masks_button = QtWidgets.QPushButton("Наложить маски")
        self.detect_text_on_video_button = QtWidgets.QPushButton("Распознать текст на видео")
        self.detect_faces_on_video_button = QtWidgets.QPushButton("Обнаружить лица на видео")
        self.apply_masks_on_faces_button = QtWidgets.QPushButton("Наложить маски на лица")
        self.load_video_button = QtWidgets.QPushButton("Загрузить видео")

        layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()

        left_layout.addWidget(self.left_label)
        left_layout.addWidget(self.load_image_button)
        left_layout.addWidget(self.detect_text_button)
        left_layout.addWidget(self.display_text_button)
        left_layout.addWidget(self.detect_text_on_video_button)
        left_layout.addWidget(self.detect_faces_on_video_button)
        left_layout.addWidget(self.apply_masks_on_faces_button)
        left_layout.addWidget(self.load_video_button)

        right_layout.addWidget(self.right_label)
        right_layout.addWidget(self.detect_faces_button)
        right_layout.addWidget(self.apply_masks_button)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.load_image_button.clicked.connect(self.load_image)
        self.detect_text_button.clicked.connect(self.detect_and_display_text)
        self.display_text_button.clicked.connect(self.display_recognized_text)
        self.detect_faces_button.clicked.connect(self.detect_and_display_faces)
        self.apply_masks_button.clicked.connect(self.apply_face_masks)
        self.detect_text_on_video_button.clicked.connect(self.detect_text_on_video)
        self.detect_faces_on_video_button.clicked.connect(self.detect_faces_on_video)
        self.apply_masks_on_faces_button.clicked.connect(self.apply_masks_on_faces)
        self.load_video_button.clicked.connect(self.load_video)

    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.show_image(self.left_label, self.image)

    def load_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать видео", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.show_image(self.left_label, frame)

    def show_image(self, label, image):
        qt_img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def detect_and_display_text(self):
        if hasattr(self, 'image'):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.show_image(self.right_label, self.image)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение.")

    def display_recognized_text(self):
        if hasattr(self, 'image'):
            pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_image)
            QtWidgets.QMessageBox.information(self, "Распознанный текст", text)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение.")

    def detect_and_display_faces(self):
        if hasattr(self, 'image'):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.show_image(self.right_label, self.image)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение.")

    def apply_face_masks(self):
        if hasattr(self, 'image'):
            mask_path = "babulech.png"
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                mask_resized = cv2.resize(mask, (w, h))
                mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2BGRA)
                roi = self.image[y:y+h, x:x+ w]
                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_resized[:, :, 3]))
                roi_fg = cv2.bitwise_and(mask_resized[:, :, :3], mask_resized[:, :, :3])
                self.image[y:y+h, x:x+w] = cv2.addWeighted(roi_bg, 1.0, roi_fg, 0.4, 0)
            self.show_image(self.right_label, self.image)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение.")

    def apply_masks_on_faces(self):
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    mask_path = "C:\\Users\\epish\\Downloads\\babulech.png"
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2BGRA)
                    roi = frame[y:y+h, x:x+w]
                    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_resized[:, :, 3]))
                    roi_fg = cv2.bitwise_and(mask_resized[:, :, :3], mask_resized[:, :, :3])
                    frame[y:y+h, x:x+w] = cv2.addWeighted(roi_bg, 1.0, roi_fg, 0.4, 0)
                self.show_image(self.right_label, frame)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите видео.")

    def detect_text_on_video(self):
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.show_image(self.right_label, frame)
                self.recognize_text(frame)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите видео.")

    def recognize_text(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil_image)
        self.display_text_message(text)

    def display_text_message(self, text):
        QtWidgets.QMessageBox.information(self, "Распознанный текст", text)

    def detect_faces_on_video(self):
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.show_image(self.right_label, frame)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала загрузите видео.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()