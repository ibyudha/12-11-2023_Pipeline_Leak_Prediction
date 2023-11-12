import os, time, threading, io
import tkinter as tk
import pandas as pd
import numpy as np
import pygame
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import ttk, filedialog, messagebox, font as tkFont
from PIL import Image, ImageTk, ImageDraw, ImageFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, mean_squared_error, r2_score
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
        self.pane_preprocessing = None
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
        self.preprocessing_btn = ttk.Button(self.menu_frame, text="Preprocessing")
        self.preprocessing_btn.grid(row=1, column=0, padx=5, pady=10, sticky="nsew")
        self.preprocessing_btn.config(command=lambda: self.show_pane(self.create_preprocessing_pane()))
        self.pelatihan_btn = ttk.Button(self.menu_frame, text="Training")
        self.pelatihan_btn.grid(row=2, column=0, padx=5, pady=10, sticky="nsew")
        self.pelatihan_btn.config(command=lambda: self.show_pane(self.create_pelatihan_pane()))
        self.prediksi_btn = ttk.Button(self.menu_frame, text="Prediction")
        self.prediksi_btn.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
        self.prediksi_btn.config(command=lambda: self.show_pane(self.create_prediksi_pane()))
        self.bantuan_btn = ttk.Button(self.menu_frame, text="Help")
        self.bantuan_btn.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")
        self.bantuan_btn.config(command=lambda: self.show_pane(self.create_other_pane("help")))
        self.about_btn = ttk.Button(self.menu_frame, text="About")
        self.about_btn.grid(row=5, column=0, padx=5, pady=10, sticky="nsew")
        self.about_btn.config(command=lambda: self.show_pane(self.create_other_pane("about")))
        self.keluar_btn = ttk.Button(self.menu_frame, text="Exit", style="Accent.TButton")
        self.keluar_btn.grid(row=6, column=0, padx=5, pady=10, sticky="nsew")
        self.keluar_btn.config(command=self.confirm_exit)
        self.show_pane(self.create_other_pane("home"))
    def create_preprocessing_pane(self):
        frame = ttk.Frame(self.paned, padding=(20, 10))
        frame.grid(column=0, row=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        label = tk.Label(frame, text="Preprocessing Page", font=('Space Age', 24), anchor="w", justify="left")
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
        preprocessing_btn = ttk.Button(submit_frame, text="Process", style="Accent.TButton")
        preprocessing_btn.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        preprocessing_btn.config(command= lambda: self.preprocessing_data_threaded(hasil_plot_frame))
        return frame
    def preprocessing_data_threaded(self, pane):
        training_thread = threading.Thread(target=lambda: self.preprocessing_data(pane))
        training_thread.start()
        loading_label = ttk.Label(pane, text="Processing, please wait...")
        loading_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        progress = ttk.Progressbar(pane, mode='indeterminate')
        progress.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        progress.start()
    def preprocessing_data(self, pane):
        csv_file = self.file_path_var.get()
        if not csv_file:
            messagebox.showerror("Error", "Please select the CSV file first.")
            return
        try:
            data = pd.read_csv(csv_file)
            df = pd.DataFrame(data)
            transposed_df = df.pivot_table(index=['leak_size', 'leak_location'], columns='length', values=['flow_rate', 'pressure'], aggfunc='first')
            if not os.path.exists("preprocessed") : 
                os.makedirs("preprocessed")
            transposed_df.to_csv('preprocessed/transposed_data.csv')
            plot_data = pd.read_csv('preprocessed/transposed_data.csv')
            plot_data = plot_data.iloc[2:, 1:]
            plot_data.columns = ['leak_location'] + [f'length_{col}' for col in plot_data.columns[1:]]
            plot_data.reset_index(drop=True, inplace=True)
            if not os.path.exists("dataset") : 
                os.makedirs("dataset/train")
                os.makedirs("dataset/test")
            for i in range(5):
                split_margin = int(len(plot_data)*(50+(i*10))/100)
                trainset = plot_data.iloc[:split_margin, :]
                testset = plot_data.iloc[split_margin:, :]
                trainset.to_csv(f'dataset/train/train_{50 + (i * 10)}_{50 - (i * 10)}.csv', index=False)
                testset.to_csv(f'dataset/test/test_{50 + (i * 10)}_{50 - (i * 10)}.csv', index=False)
            if not os.path.exists("visualization/preprocessing") : 
                os.makedirs("visualization/preprocessing")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
            for j in range(int(len(plot_data)/22)):
                for i in range(22):
                    flowrate = plot_data.iloc[i+(j*22):i+(j*22)+1, 1:4].T
                    pressure = plot_data.iloc[i+(j*22):i+(j*22)+1, 4:].T
                    length = data[0:3]['length']
                    ax1.plot(length, flowrate, marker='o', linestyle='-', label=f'Flow Rate {i + 1}')
                    ax2.plot(length, pressure, marker='o', linestyle='-', label=f'Pressure {i + 1}')
                judul = "ALL SCENARIO"
                fig.suptitle(judul, fontsize=16)
                ax1.set_title('Flow Rate Comparison')
                ax1.set_xlabel('Length')
                ax1.set_ylabel('Flow Rate')
                ax1.grid(True)
                ax2.set_title('Pressure Comparison')
                ax2.set_xlabel('Length')
                ax2.set_ylabel('Pressure')
                ax2.grid(True)
            plt.savefig('visualization/preprocessing/all_scenario.png', bbox_inches='tight', pad_inches=0.1)
            plt.show()
            progress = pane.grid_slaves(row=4, column=0)[0]
            loading_label = pane.grid_slaves(row=3, column=0)[0]
            progress.stop()
            progress.grid_forget()
            loading_label.grid_forget()
            messagebox.showinfo("Process Complete!", "Data has been preprocessed.")
            if self.pane_preprocessing:
                for widget in self.pane_preprocessing.winfo_children():
                    widget.destroy() 
        except Exception as e:
            messagebox.showerror("Error", str(e))
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
        training_thread = threading.Thread(target=lambda: self.train_model(pane))
        training_thread.start()
        loading_label = ttk.Label(pane, text="Processing, please wait...")
        loading_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        progress = ttk.Progressbar(pane, mode='indeterminate')
        progress.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        progress.start()
    def train_model(self, pane):
        csv_file = self.file_path_var.get()
        if not csv_file:
            messagebox.showerror("Error", "Please select the CSV file first.")
            return
        try:
            data = pd.read_csv(csv_file)
            X = data.drop("leak_location", axis=1)
            y = data["leak_location"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            if not os.path.exists("visualization/training/") : 
                os.makedirs("visualization/training/")
            model_reg = Sequential()
            model_reg.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
            model_reg.add(BatchNormalization())
            model_reg.add(Dense(128, activation='relu', kernel_regularizer='l2'))
            model_reg.add(Dropout(0.5))
            model_reg.add(Dense(64, activation='relu'))
            model_reg.add(Dense(1, activation='linear'))
            optimizer = RMSprop(learning_rate=0.001)
            model_reg.compile(loss='mean_squared_error', optimizer=optimizer)
            early_stopping_reg = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
            history_reg = model_reg.fit(X_train_scaled, y_train, epochs=1000, batch_size=22, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping_reg])
            predictions_ann = model_reg.predict(X_test_scaled)
            mse_ann = model_reg.evaluate(X_test_scaled, y_test)
            formatted_mse_ann = "{:.2f}".format(mse_ann)
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))
            ax[0, 0].scatter(y_test, predictions_ann, color='red', label='Predictions')
            ax[0, 0].scatter(y_test, y_test, color='blue', label='True Values')
            ax[0, 0].set_xlabel("True Values")
            ax[0, 0].set_ylabel("Predictions")
            ax[0, 0].set_title(f"ANN Regression, MSE: {formatted_mse_ann}")
            ax[0, 0].legend()
            svm_model = SVR(kernel='linear', C=1.0)
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 'scale', 'auto']}
            grid_search = GridSearchCV(svm_model, param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train_scaled, y_train)
            best_svm_model = grid_search.best_estimator_
            predictions_svm = best_svm_model.predict(X_test_scaled)
            mse_svm = mean_squared_error(y_test, predictions_svm)
            formatted_mse_svm = "{:.2f}".format(mse_svm)
            ax[0, 1].scatter(y_test, predictions_svm, color='red', label='Predictions')
            ax[0, 1].scatter(y_test, y_test, color='blue', label='True Values')
            ax[0, 1].set_xlabel("True Values")
            ax[0, 1].set_ylabel("Predictions")
            ax[0, 1].set_title(f"SVM Regression, MSE: {formatted_mse_svm}")
            ax[0, 1].legend()
            dt_model = DecisionTreeRegressor(random_state=42)
            param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            grid_search_dt = GridSearchCV(dt_model, param_grid_dt, scoring='neg_mean_squared_error', cv=5)
            grid_search_dt.fit(X_train, y_train)
            best_dt_model = grid_search_dt.best_estimator_
            predictions_dt = best_dt_model.predict(X_test)
            mse_dt = mean_squared_error(y_test, predictions_dt)
            formatted_mse_dt = "{:.2f}".format(mse_dt)
            ax[1, 0].scatter(y_test, predictions_dt, color='red', label='Predictions')
            ax[1, 0].scatter(y_test, y_test, color='blue', label='True Values')
            ax[1, 0].set_xlabel("True Values")
            ax[1, 0].set_ylabel("Predictions")
            ax[1, 0].set_title(f"Decision Tree Regression, MSE: {formatted_mse_dt}")
            ax[1, 0].legend()
            rf_model = RandomForestRegressor(random_state=42)
            param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            grid_search_rf = GridSearchCV(rf_model, param_grid_rf, scoring='neg_mean_squared_error', cv=5)
            grid_search_rf.fit(X_train, y_train)
            best_rf_model = grid_search_rf.best_estimator_
            predictions_rf = best_rf_model.predict(X_test)
            mse_rf = mean_squared_error(y_test, predictions_rf)
            formatted_mse_rf = "{:.2f}".format(mse_rf)
            ax[1, 1].scatter(y_test, predictions_rf, color='red', label='Predictions')
            ax[1, 1].scatter(y_test, y_test, color='blue', label='True Values')
            ax[1, 1].set_xlabel("True Values")
            ax[1, 1].set_ylabel("Predictions")
            ax[1, 1].set_title(f"Random Forest Regression, MSE: {formatted_mse_rf}")
            ax[1, 1].legend()
            file_name, file_extension = os.path.splitext(csv_file)
            file_name_only = os.path.basename(file_name)
            get_value_type = file_name_only
            train_name = get_value_type.split('_')[0]
            train_type = get_value_type.split('_')[1]
            fig.suptitle(f"Skenario: {train_name} {train_type}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f'visualization/training/regression_results_{file_name_only}.png', bbox_inches='tight', pad_inches=0.1)
            plt.show()
            if not os.path.exists("model") : 
                os.makedirs("model")
            model_reg.save(f'model/model-ann-{train_type}.h5')
            joblib.dump(best_svm_model, f'model/model-svm-{train_type}.joblib')
            joblib.dump(best_dt_model, f'model/model-dt-{train_type}.joblib')
            joblib.dump(best_rf_model, f'model/model-rf-{train_type}.joblib')
            progress = pane.grid_slaves(row=4, column=0)[0]
            loading_label = pane.grid_slaves(row=3, column=0)[0]
            progress.stop()
            progress.grid_forget()
            loading_label.grid_forget()
            messagebox.showinfo("Process Complete!", "Model training has been completed.")
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
        label_entry = tk.Label(entry_frame, text="(1) Pressure (psig)")
        label_entry.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pressure_1 = ttk.Spinbox(entry_frame, from_=0, to=26, increment=0.0000000000001)
        self.pressure_1.insert(0, "0.0000000000000")
        self.pressure_1.grid(row=1, column=0, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="(1) Flow Rate (kg/s)")
        label_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.flow_rate_1 = ttk.Spinbox(entry_frame, from_=0, to=4220, increment=0.01)
        self.flow_rate_1.insert(0, "0.00000000000")
        self.flow_rate_1.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="(2) Pressure (psig)")
        label_entry.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.pressure_2 = ttk.Spinbox(entry_frame, from_=0, to=26, increment=0.0000000000001)
        self.pressure_2.insert(0, "0.0000000000000")
        self.pressure_2.grid(row=3, column=0, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="(2) Flow Rate (kg/s)")
        label_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.flow_rate_2 = ttk.Spinbox(entry_frame, from_=0, to=4220, increment=0.01)
        self.flow_rate_2.insert(0, "0.00000000000")
        self.flow_rate_2.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="(3) Pressure (psig)")
        label_entry.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.pressure_3 = ttk.Spinbox(entry_frame, from_=0, to=26, increment=0.0000000000001)
        self.pressure_3.insert(0, "0.0000000000000")
        self.pressure_3.grid(row=5, column=0, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="(3) Flow Rate (kg/s)")
        label_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.flow_rate_3 = ttk.Spinbox(entry_frame, from_=0, to=4220, increment=0.01)
        self.flow_rate_3.insert(0, "0.00000000000")
        self.flow_rate_3.grid(row=5, column=1, padx=5, pady=10, sticky="w")
        label_entry = tk.Label(entry_frame, text="Load Model ML:")
        label_entry.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        model_file = ttk.Entry(entry_frame, textvariable=self.file_path_var)
        model_file.grid(row=7, column=0, padx=5, pady=10, sticky="w")
        self.browse_btn = ttk.Button(entry_frame, text="Choose Model", style="Accent.TButton")
        self.browse_btn.grid(row=7, column=1, padx=5, pady=10)
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
        file_path = filedialog.askopenfilename(filetypes=[("h5 files, joblib files", "*.h5;*.joblib")])
        self.file_path_var.set(file_path)
    def predict_leak(self):
        if self.file_path_var.get().endswith('.h5'):
            model = load_model(self.file_path_var.get())
        elif self.file_path_var.get().endswith('.joblib'):
            model = joblib.load(self.file_path_var.get())
        else:
            messagebox.showerror("Error", "Unsupported file type. Please choose a valid .h5 or .joblib file.")
            return
        pressure_1 = float(self.pressure_1.get())
        pressure_2 = float(self.pressure_2.get())
        pressure_3 = float(self.pressure_3.get())
        flow_rate_1 = float(self.flow_rate_1.get())
        flow_rate_2 = float(self.flow_rate_2.get())
        flow_rate_3 = float(self.flow_rate_3.get())
        input_data = np.array([[flow_rate_1, flow_rate_2, flow_rate_3, pressure_1, pressure_2, pressure_3]])
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
        file_name, file_extension = os.path.splitext(self.file_path_var.get())
        file_name_only = os.path.basename(file_name)
        date_text = f"Predicted at {formatted_datetime}. With: {file_name_only}"

        predictions = model.predict(input_data)
        if file_extension == ".h5":
            prediction_result = predictions[0][0]
        else:
            prediction_result = predictions[0]

        image_path = 'assets/images/PipelineLeakLocation.png'
        prediction_img = Image.open(image_path)
        image_width, image_height = prediction_img.size
        draw = ImageDraw.Draw(prediction_img)
        font = ImageFont.truetype("assets/fonts/space_age.ttf", 50)
        font_date = ImageFont.truetype("assets/fonts/space_age.ttf", 30)
        text_color = (220, 220, 220)
        text_x = (image_width) // 8
        text = f"Leakage Location around {prediction_result:.2f} m"
        text_x_date = (image_width) // 6.7
        text_y_date = (image_height) // 1.3
        text_y = (image_height) // 2
        draw.text((text_x, text_y), text, fill=text_color, font=font)
        draw.text((text_x_date, text_y_date), date_text, fill=text_color, font=font_date)
        prediction_img.show()
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
    root.title("Pipeline Leak Prediction - PLP for AIMS")
    root.iconbitmap('assets/images/aims.ico')
    root.attributes('-fullscreen', True)
    root.tk.call("source", 'assets/setting.tcl')
    root.tk.call("set_theme", "dark")
    app.grid(column=0, row=0, sticky="nsew")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app.notebook.grid_columnconfigure(0, weight=1)
    app.notebook.grid_rowconfigure(0, weight=1)
    root.mainloop()
