import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame, StringVar, Entry

from PIL import Image, ImageTk
import cv2
import numpy as np

class DarkClusterAnalyzer:
    def __init__(self, root_window):
        # --- Inicjalizacja Głównego Okna ---
        self.root = root_window
        self.root.title("Analizator Ciemnych Skupisk Pikseli")
        self.root.geometry("1200x800")

        # Zmienne do przechowywania obrazów i danych
        self.original_image = None
        self.display_image_tk = None
        self.total_image_area = 0.0
        self.highlight_color = (255, 0, 255) # Neonowy Róż/Magenta (BGR)

        # --- Tworzenie Interfejsu Graficznego ---
        
        # 1. Górny panel na kontrolki
        control_frame = Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Przycisk do wczytywania
        load_button = Button(control_frame, text="Wczytaj Obraz...", command=self.load_image)
        load_button.grid(row=0, column=0, padx=10, pady=5)

        # --- NOWA LOGIKA: Synchronizacja Suwaków i Pól Tekstowych ---
        
        # Rejestracja funkcji walidacji
        vcmd_brightness = (self.root.register(self.validate_input), '%P', 0, 255)
        vcmd_area = (self.root.register(self.validate_input), '%P', 0, 5000)

        # Kontrolki progu jasności
        Label(control_frame, text="Próg Jasności:").grid(row=0, column=1, sticky='w', padx=(20, 5))
        self.brightness_var = StringVar(value="50")
        self.brightness_var.trace_add("write", self.update_from_variable)
        
        self.brightness_slider = Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=150, showvalue=0, variable=self.brightness_var)
        self.brightness_slider.grid(row=0, column=2)
        
        self.brightness_entry = Entry(control_frame, width=5, textvariable=self.brightness_var, validate="key", validatecommand=vcmd_brightness)
        self.brightness_entry.grid(row=0, column=3, padx=5)

        # Kontrolki minimalnej wielkości skupiska
        Label(control_frame, text="Min. Wielkość Skupiska (px):").grid(row=0, column=4, sticky='w', padx=(20, 5))
        self.area_var = StringVar(value="100")
        self.area_var.trace_add("write", self.update_from_variable)

        self.area_slider = Scale(control_frame, from_=0, to=5000, orient=tk.HORIZONTAL, length=150, showvalue=0, variable=self.area_var)
        self.area_slider.grid(row=0, column=5)

        self.area_entry = Entry(control_frame, width=5, textvariable=self.area_var, validate="key", validatecommand=vcmd_area)
        self.area_entry.grid(row=0, column=6, padx=5)
        
        # Wyświetlanie procentu
        Label(control_frame, text="Zaznaczony obszar:", font=("Arial", 12)).grid(row=0, column=7, sticky='w', padx=(30, 5))
        self.percentage_var = StringVar(value="0.00%")
        self.percentage_label = Label(control_frame, textvariable=self.percentage_var, font=("Arial", 12, "bold"), fg="magenta4")
        self.percentage_label.grid(row=0, column=8)
        
        # Etykieta do wyświetlania obrazu
        self.image_label = Label(self.root)
        self.image_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

    def validate_input(self, new_value, min_val_str, max_val_str):
        """Sprawdza, czy wprowadzona wartość jest poprawną liczbą w danym zakresie."""
        if not new_value: # Pozwala na usunięcie całego tekstu
            return True
        try:
            value = int(new_value)
            min_val = int(min_val_str)
            max_val = int(max_val_str)
            if min_val <= value <= max_val:
                return True
            return False
        except ValueError:
            return False

    def load_image(self):
        """Otwiera dialog wyboru pliku i wczytuje obraz."""
        file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Wszystkie pliki", "*.*")])
        if not file_path:
            return

        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            self.total_image_area = 0.0
            return
            
        height, width, _ = self.original_image.shape
        self.total_image_area = float(height * width)
        self.percentage_var.set("0.00%")
        self.update_image_preview()

    def update_from_variable(self, *args):
        """Funkcja pośrednicząca, która uruchamia aktualizację obrazu."""
        self.update_image_preview()

    def update_image_preview(self):
        """Główna funkcja, która przetwarza obraz i aktualizuje podgląd."""
        if self.original_image is None or self.total_image_area == 0:
            return

        try:
            brightness_threshold = int(self.brightness_var.get())
            min_cluster_area = int(self.area_var.get())
        except (ValueError, tk.TclError):
            return # Ignoruj błędy, gdy pole jest tymczasowo puste

        display_image = self.original_image.copy()
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_highlighted_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_cluster_area:
                total_highlighted_area += area
                cv2.drawContours(display_image, [contour], -1, self.highlight_color, cv2.FILLED)

        percentage = (total_highlighted_area / self.total_image_area) * 100
        self.percentage_var.set(f"{percentage:.2f}%")
        
        self.display_in_tkinter(display_image)
        
    def display_in_tkinter(self, image_bgr):
        """Konwertuje obraz OpenCV i wyświetla go w etykiecie Tkinter."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        canvas_width = self.image_label.winfo_width()
        canvas_height = self.image_label.winfo_height()
        if canvas_width < 50 or canvas_height < 50:
            canvas_width, canvas_height = 1000, 700
            
        pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.display_image_tk = ImageTk.PhotoImage(image=pil_image)
        self.image_label.config(image=self.display_image_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkClusterAnalyzer(root)
    root.mainloop()