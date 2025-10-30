import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame, StringVar, Entry, messagebox
from tkinter import ttk
import cv2
import rawpy
import os

# Ta funkcja jest sercem analizy, przeniesiona z wersji bez GUI
def analyze_image_darkness(file_path, threshold, min_area):
    """Analizuje pojedynczy obraz i zwraca procentowy udział ciemnych obszarów."""
    try:
        with rawpy.imread(file_path) as raw:
            bgr_image = cv2.cvtColor(raw.postprocess(use_camera_wb=True), cv2.COLOR_RGB2BGR)
    except Exception:
        bgr_image = cv2.imread(file_path)

    if bgr_image is None:
        return None

    height, width, _ = bgr_image.shape
    total_image_area = float(height * width)
    if total_image_area == 0: return 0.0
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_highlighted_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
            
    return (total_highlighted_area / total_image_area) * 100

class BatchAnalyzerGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Wsadowy Analizator Ciemnych Obszarów")
        self.root.geometry("800x600")

        self.input_folder_path = ""

        # --- Panel sterowania ---
        control_frame = Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        # Wybór folderu
        Button(control_frame, text="Wybierz Folder...", command=self.select_folder).grid(row=0, column=0, padx=(0, 10))
        self.folder_label_var = StringVar(value="Nie wybrano folderu")
        Label(control_frame, textvariable=self.folder_label_var, fg="blue").grid(row=0, column=1, columnspan=3, sticky='w')

        # Parametry
        vcmd_brightness = (self.root.register(self.validate_input), '%P', 0, 255)
        vcmd_area = (self.root.register(self.validate_input), '%P', 0, 999999)

        Label(control_frame, text="Próg Jasności:").grid(row=1, column=0, pady=(10,0), sticky='w')
        self.brightness_var = StringVar(value="50")
        Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, showvalue=0, variable=self.brightness_var).grid(row=1, column=1, pady=(10,0))
        Entry(control_frame, width=5, textvariable=self.brightness_var, validate="key", vcmd=vcmd_brightness).grid(row=1, column=2, pady=(10,0), padx=5)

        Label(control_frame, text="Min. Wielkość Skupiska:").grid(row=2, column=0, sticky='w')
        self.area_var = StringVar(value="100")
        Scale(control_frame, from_=0, to=5000, orient=tk.HORIZONTAL, showvalue=0, variable=self.area_var).grid(row=2, column=1)
        Entry(control_frame, width=7, textvariable=self.area_var, validate="key", vcmd=vcmd_area).grid(row=2, column=2, padx=5)
        
        # Przycisk start
        Button(control_frame, text="▶ Start Analizy", font=("Arial", 12, "bold"), bg="pale green", command=self.start_analysis).grid(row=0, column=4, rowspan=3, padx=(50, 0), ipady=10, ipadx=10)

        # --- Panel wyników (Tabela) ---
        results_frame = Frame(self.root, padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_tree = ttk.Treeview(results_frame, columns=("file", "percentage"), show="headings")
        self.results_tree.heading("file", text="Nazwa Pliku")
        self.results_tree.heading("percentage", text="Zaznaczony Obszar (%)")
        self.results_tree.column("file", width=400)
        self.results_tree.column("percentage", width=150, anchor='e')

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_folder(self):
        path = filedialog.askdirectory(title="Wybierz folder ze zdjęciami")
        if path:
            self.input_folder_path = path
            self.folder_label_var.set(f"Wybrano: ...{path[-40:]}") # Pokaż końcówkę ścieżki

    def start_analysis(self):
        if not self.input_folder_path:
            messagebox.showerror("Błąd", "Proszę najpierw wybrać folder do analizy.")
            return

        # Wyczyść poprzednie wyniki
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Pobierz parametry z GUI
        try:
            threshold = int(self.brightness_var.get())
            min_area = int(self.area_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Błąd", "Wprowadzone parametry są nieprawidłowe.")
            return

        # Znajdź pliki i rozpocznij analizę
        supported_ext = ['.nef', '.cr2', '.arw', '.dng', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        files_to_process = sorted([f for f in os.listdir(self.input_folder_path) if os.path.splitext(f)[1].lower() in supported_ext])

        if not files_to_process:
            messagebox.showinfo("Informacja", "W wybranym folderze nie znaleziono obsługiwanych plików.")
            return
            
        for filename in files_to_process:
            full_path = os.path.join(self.input_folder_path, filename)
            result = analyze_image_darkness(full_path, threshold, min_area)
            
            if result is not None:
                result_str = f"{result:.2f}%"
                self.results_tree.insert("", "end", values=(filename, result_str))
            else:
                self.results_tree.insert("", "end", values=(filename, "Błąd odczytu"))
            
            # Odśwież interfejs, aby wyniki pojawiały się na bieżąco
            self.root.update_idletasks()
            
        messagebox.showinfo("Koniec", f"Analiza została zakończona.\nPrzetworzono {len(files_to_process)} plików.")

    def validate_input(self, new_val, min_v, max_v):
        if not new_val: return True
        try:
            val = int(new_val)
            return int(min_v) <= val <= int(max_v)
        except ValueError:
            return False

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchAnalyzerGUI(root)
    root.mainloop()