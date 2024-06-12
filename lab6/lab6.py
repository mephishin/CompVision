import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from functools import partial

class MotionDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Приложение для обнаружения движения")
        
        self.video_source = None
        self.capture = None
        self.frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=32, detectShadows=True)
        
        self.original_label = tk.Label(master)
        self.original_label.grid(row=0, column=0)
        
        self.processed_label = tk.Label(master)
        self.processed_label.grid(row=0, column=1)
        
        self.webcam_button = tk.Button(master, text="Web-камера", command=self.start_webcam_motion_detection)
        self.webcam_button.grid(row=1, column=0)
        
        self.video_button = tk.Button(master, text="Видео file", command=self.start_video_motion_detection)
        self.video_button.grid(row=1, column=1)
        
        self.stop_button = tk.Button(master, text="Стоп", command=self.stop_motion_detection)
        self.stop_button.grid(row=2, column=0, columnspan=2, pady=5)
        self.stop_button.config(state="disabled")
        
    def start_webcam_motion_detection(self):
        self.video_source = 0 # 0 for webcam
        self.capture = cv2.VideoCapture(self.video_source)
        self.stop_button.config(state="normal")
        self.webcam_button.config(state="disabled")
        self.video_button.config(state="disabled")
        self.motion_detection()
        
    def start_video_motion_detection(self):
        self.video_source = filedialog.askopenfilename() # Select video file
        self.capture = cv2.VideoCapture(self.video_source)
        self.stop_button.config(state="normal")
        self.webcam_button.config(state="disabled")
        self.video_button.config(state="disabled")
        self.motion_detection()
        
    def stop_motion_detection(self):
        if self.capture is not None:
            self.capture.release()
            self.stop_button.config(state="disabled")
            self.webcam_button.config(state="normal")
            self.video_button.config(state="normal")
        
    def motion_detection(self):
        ret, frame = self.capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.background_subtractor.apply(gray_frame)
            fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)[1]
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 700:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(fg_mask)
            
            original_image = ImageTk.PhotoImage(image=Image.fromarray(original_image))
            processed_image = ImageTk.PhotoImage(image=processed_image)
            
            self.original_label.config(image=original_image)
            self.original_label.image = original_image
            
            self.processed_label.config(image=processed_image)
            self.processed_label.image = processed_image
            
            self.master.after(10, self.motion_detection)
        else:
            self.stop_motion_detection()

def main():
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
