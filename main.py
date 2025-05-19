import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import neurokit2 as nk
from catboost import CatBoostClassifier
from scipy import interpolate
import pyedflib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========================
# 1. Функции обработки данных
# ========================

def load_edf_file(file_path):
    """Загрузка EDF-файла и извлечение ЭКГ сигнала"""
    try:
        with pyedflib.EdfReader(file_path) as file:
            ecg_channels = [i for i, label in enumerate(file.getSignalLabels())
                           if 'ECG' in label.upper() or 'EKG' in label.upper()]
            
            if not ecg_channels:
                raise ValueError("ECG channel not found")
            
            ecg_signal = file.readSignal(ecg_channels[0])
            sampling_rate = file.getSampleFrequency(ecg_channels[0])
            return ecg_signal, sampling_rate
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None, None

def process_ecg_signal(signal, sampling_rate, num_points=250):
    """Обработка ЭКГ сигнала и извлечение кардиоциклов"""
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]
        
        if len(rpeaks) < 2:
            return []
        
        processed_cycles = []
        
        for i in range(1, len(rpeaks)):
            prev_r = rpeaks[i-1]
            current_r = rpeaks[i]
            cycle_length = current_r - prev_r
            cycle_start = current_r - cycle_length//2
            cycle_end = current_r + cycle_length//2
            
            if cycle_start < 0 or cycle_end >= len(cleaned):
                continue
                
            cycle = cleaned[cycle_start:cycle_end]
            
            if len(cycle) < 10:
                continue
                
            x_orig = np.linspace(0, 1, len(cycle))
            x_new = np.linspace(0, 1, num_points)
            f = interpolate.interp1d(x_orig, cycle, kind='linear')
            interpolated = f(x_new)
            
            processed_cycles.append(interpolated)
            
        return processed_cycles
    except Exception as e:
        print(f"Signal processing error: {str(e)}")
        return []

def prepare_dataset(normal_folder, abnormal_folder):
    """Подготовка датасета"""
    X, y = [], []
    
    # Обработка нормальных примеров
    for file in os.listdir(normal_folder):
        if file.endswith('.edf'):
            signal, sr = load_edf_file(os.path.join(normal_folder, file))
            cycles = process_ecg_signal(signal, sr)
            X.extend(cycles)
            y.extend([0]*len(cycles))
    
    # Обработка аномальных примеров
    for file in os.listdir(abnormal_folder):
        if file.endswith('.edf'):
            signal, sr = load_edf_file(os.path.join(abnormal_folder, file))
            cycles = process_ecg_signal(signal, sr)
            X.extend(cycles)
            y.extend([1]*len(cycles))
    
    return pd.DataFrame(X), np.array(y)

# ========================
# 2. Система сортировки файлов
# ========================

def sort_files(source_dir="PulmHypert"):
    """Автоматическая сортировка файлов"""
    target_dirs = {
        '30': 'SDLA30',
        '50': 'SDLA50'
    }
    
    # Создание целевых директорий
    for d in target_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Поиск и сортировка файлов
    pattern = re.compile(r'.*?(\d{2}).*?\.edf')
    sorted_count = 0
    
    for filename in os.listdir(source_dir):
        if filename.endswith('.edf'):
            match = pattern.search(filename)
            if match:
                key = match.group(1)
                if key in target_dirs:
                    src = os.path.join(source_dir, filename)
                    dest = os.path.join(target_dirs[key], filename)
                    shutil.move(src, dest)
                    sorted_count += 1
                    
    print(f"Sorted {sorted_count} files")
    return sorted_count

# ========================
# 3. Модель машинного обучения
# ========================

class PHClassifier:
    def __init__(self):
        self.model = None
        self.sampling_rate = None
        
    def train_model(self):
        """Обучение модели"""
        try:
            X, y = prepare_dataset('SDLA30', 'SDLA50')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                eval_metric='AUC',
                verbose=50
            )
            
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test))
            self.model.save_model('ph_model.cbm')
            
            # Генерация отчета
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            return report
            
        except Exception as e:
            return f"Training error: {str(e)}"

# ========================
# 4. Графический интерфейс
# ========================

class PHGUI:
    def __init__(self, master):
        self.master = master
        self.classifier = PHClassifier()
        self.loaded_model = False
        
        # Настройка интерфейса
        master.title("Pulmonary Hypertension Analyzer")
        master.geometry("800x600")
        
        # Стилизация
        self.style = ttk.Style()
        self.style.configure('Main.TFrame', background='#f0f0f0')
        self.style.configure('Title.TLabel', 
                           font=('Arial', 16, 'bold'),
                           background='#f0f0f0',
                           foreground='#2c3e50')
                           
        # Основные элементы
        self.main_frame = ttk.Frame(master, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.title_label = ttk.Label(self.main_frame, 
                                   text="ECG Analysis System",
                                   style='Title.TLabel')
        self.title_label.pack(pady=20)
        
        # Блок выбора файла
        self.file_frame = ttk.Frame(self.main_frame)
        self.file_frame.pack(pady=10)
        
        self.file_label = ttk.Label(self.file_frame, text="Select EDF file:")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.file_entry = ttk.Entry(self.file_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        
        self.browse_btn = ttk.Button(self.file_frame, 
                                    text="Browse", 
                                    command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Блок результатов
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(pady=20)
        
        self.result_label = ttk.Label(self.result_frame, 
                                    text="Analysis Result:",
                                    font=('Arial', 12, 'bold'))
        self.result_label.pack(pady=10)
        
        self.result_text = tk.Text(self.result_frame,
                                 height=8,
                                 width=60,
                                 font=('Arial', 10),
                                 state=tk.DISABLED)
        self.result_text.pack()
        
        # Кнопка анализа
        self.analyze_btn = ttk.Button(self.main_frame,
                                    text="Start Analysis",
                                    command=self.analyze)
        self.analyze_btn.pack(pady=20)
        
        # Загрузка модели
        self.load_model()
        
    def load_model(self):
        """Загрузка сохраненной модели"""
        try:
            if os.path.exists('ph_model.cbm'):
                self.classifier.model = CatBoostClassifier().load_model('ph_model.cbm')
                self.loaded_model = True
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Model loaded successfully!")
                self.result_text.config(state=tk.DISABLED)
            else:
                messagebox.showwarning("Warning", "Model not found! Train first.")
        except Exception as e:
            messagebox.showerror("Error", f"Model loading failed: {str(e)}")
            
    def browse_file(self):
        """Выбор файла через диалоговое окно"""
        filepath = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, filepath)
        
    def analyze(self):
        """Выполнение анализа"""
        if not self.loaded_model:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        filepath = self.file_entry.get()
        if not filepath:
            messagebox.showwarning("Warning", "Please select a file first!")
            return
            
        try:
            # Загрузка и обработка файла
            signal, sr = load_edf_file(filepath)
            cycles = process_ecg_signal(signal, sr)
            
            if not cycles:
                messagebox.showerror("Error", "No valid ECG cycles detected")
                return
                
            # Предсказание
            X = pd.DataFrame(cycles)
            predictions = self.classifier.model.predict(X)
            probabilities = self.classifier.model.predict_proba(X)[:, 1]
            
            # Расчет среднего значения
            avg_prob = np.mean(probabilities)
            diagnosis = "SDLA50 (PH Detected)" if avg_prob > 0.5 else "SDLA30 (No PH)"
            
            # Отображение результатов
            result_text = f"""Заключение: {diagnosis}
Вероятность ЛГ: {avg_prob*100:.2f}%
Всего циклов: {len(cycles)}
Confidence Distribution:
- PH Probability > 50%: {(probabilities > 0.5).sum()} cycles
- PH Probability <= 50%: {(probabilities <= 0.5).sum()} cycles"""
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

# ========================
# Главный скрипт
# ========================

if __name__ == "__main__":
    # Шаг 1: Сортировка файлов
    print("Starting file sorting...")
    sorted_files = sort_files()
    print(f"Sorted {sorted_files} files\n")
    
    # Шаг 2: Обучение модели
    print("Starting model training...")
    classifier = PHClassifier()
    report = classifier.train_model()
    print("\nTraining report:")
    print(report)
    
    # Шаг 3: Запуск GUI
    root = tk.Tk()
    app = PHGUI(root)
    root.mainloop()