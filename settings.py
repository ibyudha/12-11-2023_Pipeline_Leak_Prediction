import os, time, threading, io
import tkinter as tk
import pandas as pd
import numpy as np
import pygame
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import ttk, filedialog, messagebox, font as tkFont
from PIL import Image, ImageTk, ImageDraw, ImageFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, load_model


class App(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.pane_pelatihan = None
        pygame.mixer.init()
        pygame.mixer.music.load('assets/sounds/BackgroundMusic.mp3')
        pygame.mixer.music.play(-1)
        parent.grid_columnconfigure(0, weight=10)
        parent.grid_columnconfigure(1, weight=90)
        self.current_pane = None
        self.setup_layouts()
    def setup_layouts(self):
        self.menu_frame = ttk.LabelFrame(self, text="Menu Aplikasi", padding=(20, 10))
        self.menu_frame.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        self.paned = ttk.PanedWindow(self)
        self.paned.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.welcome_btn = ttk.Button(self.menu_frame, text="Home")
        self.welcome_btn.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        self.welcome_btn.config(command=lambda: self.show_pane(self.create_other_pane("home")))
        self.pelatihan_btn = ttk.Button(self.menu_frame, text="Training")
        self.pelatihan_btn.grid(row=1, column=0, padx=5, pady=10, sticky="nsew")
        self.pelatihan_btn.config(command=lambda: self.show_pane(self.create_pelatihan_pane()))
        self.prediksi_btn = ttk.Button(self.menu_frame, text="Prediction")
        self.prediksi_btn.grid(row=2, column=0, padx=5, pady=10, sticky="nsew")
        self.prediksi_btn.config(command=lambda: self.show_pane(self.create_prediksi_pane()))
        self.bantuan_btn = ttk.Button(self.menu_frame, text="Help")
        self.bantuan_btn.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
        self.bantuan_btn.config(command=lambda: self.show_pane(self.create_other_pane("help")))
        self.about_btn = ttk.Button(self.menu_frame, text="About")
        self.about_btn.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")
        self.about_btn.config(command=lambda: self.show_pane(self.create_other_pane("about")))
        self.keluar_btn = ttk.Button(self.menu_frame, text="Exit", style="Accent.TButton")
        self.keluar_btn.grid(row=5, column=0, padx=5, pady=10, sticky="nsew")
        self.keluar_btn.config(command=self.confirm_exit)
        self.show_pane(self.create_other_pane("home"))
    def create_pelatihan_pane(self):
        frame = ttk.Frame(self.paned, padding=(20, 10))
        frame.grid(column=0, row=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        label = tk.Label(frame, text="Training Page", font=('Space Age', 24), anchor="w", justify="left")
        label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        csv_frame = ttk.Frame(frame)
        csv_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        label_entry = tk.Label(csv_frame, text="Load File .csv:")
        label_entry.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        entry = ttk.Entry(csv_frame, textvariable=self.file_path_var)
        entry.grid(row=1, column=0, padx=5, pady=10, sticky="w")
        browse_btn = ttk.Button(csv_frame, text="Choose File", style="Accent.TButton")
        browse_btn.grid(row=1, column=4, padx=5, pady=10)
        browse_btn.config(command=self.browse_file)
        submit_frame = ttk.Frame(frame)
        submit_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="white", background="blue")
        hasil_plot_frame = ttk.Frame(frame)
        hasil_plot_frame.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        pelatihan_btn = ttk.Button(submit_frame, text="Process", style="Accent.TButton")
        pelatihan_btn.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        pelatihan_btn.config(command= lambda: self.train_model_threaded(hasil_plot_frame))
        return frame
    def train_model_threaded(self, pane):
        training_thread = threading.Thread(target=lambda: self.train_model(pane, 'leak_size', 'ann_model_for_leak_size'))
        training_thread.start()
        loading_label = ttk.Label(pane, text="Processing, please wait...")
        loading_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        progress = ttk.Progressbar(pane, mode='indeterminate')
        progress.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        progress.start()
        training_thread = threading.Thread(target=lambda: self.train_model(pane, 'leak_location', 'ann_model_for_leak_location'))
        training_thread.start()
        loading_label = ttk.Label(pane, text="Processing, please wait...")
        loading_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        progress = ttk.Progressbar(pane, mode='indeterminate')
        progress.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        progress.start()
    def train_model(self, pane, target_column, model_name):
        csv_file = self.file_path_var.get()
        if not csv_file:
            messagebox.showerror("Error", "Please select the CSV file first.")
            return
        try:
            data = pd.read_csv(csv_file)
            X = data[['length', 'flow_rate', 'velocity', 'pressure']]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = Sequential()
            model.add(Dense(32, input_dim=4, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
            model.fit(X_train, y_train, epochs=1000, batch_size=40, validation_data=(X_test, y_test), callbacks=[early_stopping])
            mse = model.evaluate(X_test, y_test, verbose=0)
            print(f'Mean Squared Error (MSE) for {target_column}: {mse}')
            if not os.path.exists("model") : 
                os.makedirs("model")
            model.save(f'model/{model_name}.h5')
            progress = pane.grid_slaves(row=4, column=0)[0]
            loading_label = pane.grid_slaves(row=3, column=0)[0]
            progress.stop()
            progress.grid_forget()
            loading_label.grid_forget()
            if target_column == "leak_location": messagebox.showinfo("Process Complete!", "Model training has been completed.")
            if self.pane_pelatihan:
                for widget in self.pane_pelatihan.winfo_children():
                    widget.destroy() 
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def show_pane(self, pane):
            if self.current_pane: self.current_pane.grid_forget()
            if pane: pane.grid(column=0, row=0, sticky="nsew")
            self.current_pane = pane
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.file_path_var.set(file_path)
    def create_prediksi_pane(self):
        frame = ttk.Frame(self.paned, padding=(20, 10))
        frame.grid(column=0, row=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        title_frame = ttk.Frame(frame)
        title_frame.pack(pady=10, anchor="w")
        label = tk.Label(title_frame, text="Prediction Page", font=("Space Age", 24))
        label.grid(row=0, column=0, sticky="w")
        entry_frame = ttk.Frame(frame)
        entry_frame.pack(pady=10, anchor="w")
        label_entry = tk.Label(entry_frame, text="Pressure (psig)")
        label_entry.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pressure = ttk.Spinbox(entry_frame, from_=0, to=70, increment=0.001)
        self.pressure.insert(0, "0.000")
        self.pressure.grid(row=1, column=0, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="Length (m)")
        label_entry.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.length = ttk.Spinbox(entry_frame, from_=0, to=1300, increment=0.1)
        self.length.insert(0, "0.0")
        self.length.grid(row=3, column=0, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="Velocity (m/s)")
        label_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.velocity = ttk.Spinbox(entry_frame, from_=0, to=80, increment=0.01)
        self.velocity.insert(0, "0.00")
        self.velocity.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="Flow Rate (kg/s)")
        label_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.flow_rate = ttk.Spinbox(entry_frame, from_=0, to=7, increment=0.001)
        self.flow_rate.insert(0, "0.000")
        self.flow_rate.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        self.prediction_type_var = tk.StringVar()
        self.prediction_type_var.set("Leak Size")
        prediction_type_label = tk.Label(entry_frame, text="Select Prediction Type:")
        prediction_type_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        prediction_type_menu = ttk.OptionMenu(entry_frame, self.prediction_type_var, "Leak Size", "Leak Size", "Leak Location")
        prediction_type_menu.grid(row=4, column=1, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="Load Model ML:")
        label_entry.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        model_file = ttk.Entry(entry_frame, textvariable=self.file_path_var)
        model_file.grid(row=6, column=0, padx=5, pady=10, sticky="w")
        self.browse_btn = ttk.Button(entry_frame, text="Choose Model", style="Accent.TButton")
        self.browse_btn.grid(row=6, column=1, padx=5, pady=10)
        self.browse_btn.config(command=self.browse_model)
        submit_frame = ttk.Frame(frame)
        submit_frame.pack(pady=10, anchor="w")
        style = ttk.Style()
        style.configure("Accent.TButton")
        self.prediksi_btn = ttk.Button(submit_frame, text="Process", style="Accent.TButton")
        self.prediksi_btn.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.prediksi_btn.config(command=self.predict_leak)
        return frame
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("h5 files", "*.h5")])
        self.file_path_var.set(file_path)
    def predict_leak(self):
        pressure = float(self.pressure.get())
        length = float(self.length.get())
        velocity = float(self.velocity.get())
        flow_rate = float(self.flow_rate.get())
        model_file = self.file_path_var.get()
        if model_file : model = load_model(model_file)
        else:
            messagebox.showerror("Error", "Please select a model file.")
            return
        input_data = np.array([[pressure, length, velocity, flow_rate]])
        prediction_type = self.prediction_type_var.get()
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
        date_text = f"Predicted at {formatted_datetime}."
        font_date = ImageFont.truetype("assets/fonts/space_age.ttf", 30)
        if prediction_type == "Leak Size":
            predictions = model.predict(input_data)
            prediction_result = predictions[0][0]
            image_path = 'assets/images/PipelineLeakSize.png'
            prediction_img = Image.open(image_path)
            image_width, image_height = prediction_img.size
            draw = ImageDraw.Draw(prediction_img)
            font = ImageFont.truetype("assets/fonts/space_age.ttf", 50)
            text_color = (220, 220, 220)
            text_x = (image_width) // 5.3
            text = f"Leakage Size around {prediction_result:.2f}%"
        elif prediction_type == "Leak Location":
            predictions = model.predict(input_data)
            prediction_result = predictions[0][0]
            image_path = 'assets/images/PipelineLeakLocation.png'
            prediction_img = Image.open(image_path)
            image_width, image_height = prediction_img.size
            draw = ImageDraw.Draw(prediction_img)
            font = ImageFont.truetype("assets/fonts/space_age.ttf", 50)
            text_color = (220, 220, 220)
            text_x = (image_width) // 8
            text = f"Leakage Location around {prediction_result:.2f} m"
        text_x_date = (image_width) // 3.7
        text_y_date = (image_height) // 1.3
        text_y = (image_height) // 2
        draw.text((text_x, text_y), text, fill=text_color, font=font)
        draw.text((text_x_date, text_y_date), date_text, fill=text_color, font=font_date)
        prediction_img.show()
        #messagebox.showinfo("Prediction Result", text)
    def create_other_pane(self, option):
        frame = ttk.Frame(self.paned, padding=(20, 10))
        frame.grid(column=0, row=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        if option == "home": image = Image.open('assets/images/MainMenu.png')
        if option == "help": image = Image.open('assets/images/HelpMenu.png')
        if option == "about": image = Image.open('assets/images/AboutMenu.png')
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width_percent = 80 
        height_percent = 90 
        new_width = int(screen_width * (width_percent / 100))
        new_height = int(screen_height * (height_percent / 100))
        image = image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(frame, image=self.photo)
        image_label.pack()
        return frame
    def confirm_exit(self):
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to leave?")
        if confirm:
            pygame.mixer.music.stop()
            root.destroy() 
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    root.title("Pipeline Leak Prediction - PLP for AIMS")
    root.attributes('-fullscreen', True)
    root.tk.call("source", 'assets/setting.tcl')
    root.tk.call("set_theme", "dark")
    app.grid(column=0, row=0, sticky="nsew")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app.notebook.grid_columnconfigure(0, weight=1)
    app.notebook.grid_rowconfigure(0, weight=1)
    root.mainloop()
