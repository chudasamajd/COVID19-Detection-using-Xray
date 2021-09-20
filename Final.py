import eel
import test

from tkinter import Frame, Tk, BOTH, Label, Menu, filedialog, messagebox
from PIL import Image, ImageTk

import os
import codecs

def loadImageFromDialog(filename):
    filename = filename.split('\\')[1]
    loadImageFromName('D:/Python Projects/COVID19/chest-xray-pneumonia/chest_xray/test/NORMAL/'+filename)

def loadImageFromName(filename):
    # load = Image.open(filename)
    # render = ImageTk.PhotoImage(load)
    # img = Label(image=render)
    # img.image = render
    # img.place(x=(int(screenWidth)/2)-load.width/2, y=((int(screenHeight)/2))-load.height/2)
    #outputContent = "#############################################\n" + filename+"\n\n"
    outputContent = test.doOnlineInference (filename)
    #print(outputContent)
    #messagebox.showinfo(title=windowTitle + " : Result ", message=outputContent)
    eel.say_hello_js(outputContent)

eel.init('D:/Python Projects/COVID19/web', allowed_extensions=['.js', '.html'])


@eel.expose                         # Expose this function to Javascript
def say_hello_py(x):
    print(x)
    loadImageFromDialog(x)

#say_hello_py('Python World!')
#eel.say_hello_js('Python World!')   # Call a Javascript function

eel.start('UI.html')             # Start (this blocks and enters loop)



