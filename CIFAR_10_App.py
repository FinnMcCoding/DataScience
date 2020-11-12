import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import tkinter.filedialog
import tensorflow as tf

# Import model
model = tf.keras.models.load_model('cifar_model.h5')

# Make reference dictionary
classes = {
    0: 'Aeroplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

# Lay down tkinter base window
top = tk.Tk()
top.geometry('800x600')
top.title('Image Classification CIFAR10')
top.configure(background= '#CDCDCD')

# Add a heading
heading = tk.Label(top, text='Image Classification CIFAR10', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

# Command for uploading image from computer
def upload_image():
    file_path = tk.filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    label.configure(text=' ')
    show_classify_button(file_path)

# Make tkinter button
def show_classify_button(file_path):
    classify_button = tk.Button(top, text='Classify Image', command= classify(file_path), padx=10, pady = 5)
    classify_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_button.place(relx=0.79, rely=0.46)

# Preprocess uploaded image and run it through the model
def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = np.argmax(model.predict(image), axis=-1)
    sign = classes[pred[0]]

    label.configure(foreground='#011638', text=sign)

# Adding upload button
upload = tk.Button(top, text='Upload an Image', command=upload_image, padx=10, pady=5)
upload.configure(background= '#364146', foreground='white', font=('arial', 10,'bold'))
upload.pack(side='bottom', pady= 50)

sign_image = tk.Label(top)
sign_image.pack(side='bottom', expand=True)

label = tk.Label(top, background='#CDCDCD', font= ('arial', 15, 'bold'))
label.pack(side='bottom', expand=True)

top.mainloop()