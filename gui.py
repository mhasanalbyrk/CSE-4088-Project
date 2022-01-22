import tkinter as tk
from tkinter import W, mainloop

import numpy as np
import torch
import win32gui
from PIL import ImageGrab

from modelUtil import predict_local_2, create_model, predict_local_2_normal

# model = load_model('mnist.h5'
model = create_model()
model.load_state_dict(torch.load('model/model.pt'))
device = torch.device("cpu")
num = 1


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.wm_title("Handwritten Digit Recognizer")
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw Digit", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Guess", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        path = "imgs/" + "test" + ".jpg"
        im.save(path)
        pred = predict_local_2(path, model, device)
        pred2 = predict_local_2_normal(path, model, device)
        # num = num + 1
        pred_idx = np.argmax(pred)
        pred_idx2 = np.argmax(pred2)
        print(f'Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %')
        print(f'Predicted normal : {pred_idx2}, Prob: {pred2[0][pred_idx2] * 100} %')

        self.label.configure(text=str(pred_idx2) + ', ' + str(int(pred2[0][pred_idx2] * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 22
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
