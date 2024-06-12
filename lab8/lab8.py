import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Функция для чтения изображений и их меток из заданной директории
def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    images = []
    labels = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        images.append(np.array(image))
        # Извлекаем метку из имени файла
        label = int(os.path.splitext(os.path.basename(image_path))[0].split("subject")[1])
        labels.append(label)
    return images, labels


# Функция для обнаружения и отображения лиц на изображении
def recognize_and_display_face(image, image_path, output_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Создаем новый рисунок
    fig, ax = plt.subplots(figsize=(10, 10))

    # Отображаем исходное изображение
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Рисуем прямоугольники вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    output_file = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '_processed.jpg')
    plt.savefig(output_file)
    plt.close(fig)


# Путь к файлу с каскадом Хаара для обнаружения лиц
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Путь к директории с изображениями для обучения
input_path = r'images'

# Получаем изображения и их метки для обучения
images, labels = get_images(input_path)
print("Starting")

# Обучаем распознаватель
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
print("Finished. Successfully find faces.")

# Путь к директории, куда будут сохранены обработанные изображенияы
output_path = r'recognized_faces'

# Создаем директорию, если она не существует
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Выводим обработанные изображения
for image_path in os.listdir(input_path):
    if image_path.endswith('.jpg'):
        image = cv2.imread(os.path.join(input_path, image_path))
        recognize_and_display_face(image, image_path, output_path)

print(
    "Successfully handled all images and saved them with suffix _processed.jpg in package recognized_faces.")
