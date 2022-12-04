# %%
from tensorflow.keras.models import load_model
import numpy as np

import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from skimage.transform import resize
from skimage.color import rgb2hsv
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

class Root(Tk):
    def __init__(self,model):
        super(Root, self).__init__()
        self.title("Malaria Detection")
        self.minsize(640, 400)
        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        self.button()
        self.button1() 
        self.model = model
        self.label1 : Label= None
 
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)
  
    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("png files","*.png"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)
 
    def button1(self):
        self.button = ttk.Button(self.labelFrame, text = "submit", command = self.get_prediction)
        self.button.grid(column = 1, row = 20)
        
    def get_prediction(self):
        
        # img = Image.open(self.filename)
        # ph = ImageTk.PhotoImage(img)
        if self.label1 is not None:
            self.label1.config(image = "")

        img = Image.open(self.filename)
        ph = ImageTk.PhotoImage(img)

        self.label1 = Label(image = ph)
        self.label1.image = ph
        self.label1.place(x = 260, y = 120)

        my_image = image.load_img(self.filename)  #input 
        my_image = image.img_to_array(my_image)
        my_image = rgb2hsv(resize(my_image, (25, 25)))[..., 1]
        my_image = np.expand_dims(my_image, axis=0)
        s=self.model.predict(my_image, )
        #print(np.argmax(s[0]))
        if(np.argmax(s[0]) > 0.5):
            self.label.configure(text="Parasitized")
        else:
            self.label.configure(text="Uninfected") 

        
cnn = load_model(filepath = 'CNN_HSV_Best.tf')
root = Root(cnn)
root.mainloop()


