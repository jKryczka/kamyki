import cv2
import numpy as np
import rawpy
import os
from pillow_heif import register_heif_opener

# --- Konfiguracja ---
INPUT_FOLDER = 'foty'
OUTPUT_FOLDER = 'znormalizowane_i_wyrownane' # Zmieniona nazwa folderu wyjściowego
SUPPORTED_EXTENSIONS = ['.nef', '.cr2', '.arw', '.dng', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
# --- NOWY PARAMETR ---
# Ustaw docelową średnią jasność dla wszystkich obrazów (w skali 0-255).
# Wartość 128 to neutralny, środkowy poziom szarości.
TARGET_AVG_BRIGHTNESS = 128

def reduce_shadows_and_normalize(image_path): # Zmieniono nazwę funkcji dla jasności
    """
    Wczytuje obraz, redukuje cienie, normalizuje jasność i zwraca przetworzony obraz.
    """
    filename = os.path.basename(image_path)
    
    # Krok 1: Wczytaj obraz
    try:
        with rawpy.imread(image_path) as raw:
            rgb_image = raw.postprocess(use_camera_wb=True)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        print(f"INFO: Przetwarzam plik RAW: {filename}")
    except Exception:
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            print(f"WARNING: Pominięto plik. Nie można go wczytać: {filename}")
            return None
        print(f"INFO: Przetwarzam standardowy obraz: {filename}")

    # Krok 2: Konwertuj obraz do przestrzeni barw LAB
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Krok 3: Usuwanie cieni (tak jak poprzednio)
    l_channel_float = l_channel.astype(float) / 255.0
    blur_radius = 101
    blurred_l = cv2.GaussianBlur(l_channel_float, (blur_radius, blur_radius), 0)
    corrected_l_float = l_channel_float / (blurred_l + 1e-6)
    corrected_l_float = cv2.normalize(corrected_l_float, None, 0, 255, cv2.NORM_MINMAX)
    corrected_l = corrected_l_float.astype(np.uint8)

    # --- POCZĄTEK MODYFIKACJI ---
    # Krok 4: Normalizacja jasności do docelowej wartości średniej
    
    # Oblicz aktualną średnią jasność kanału L po usunięciu cieni
    current_avg_l = np.mean(corrected_l)
    
    # Oblicz różnicę, którą należy dodać do każdego piksela
    diff_l = TARGET_AVG_BRIGHTNESS - current_avg_l
    
    # Zastosuj korektę. Używamy float32, aby uniknąć błędów przy dodawaniu.
    normalized_l_float = corrected_l.astype(np.float32) + diff_l
    
    # Ogranicz wartości do prawidłowego zakresu 0-255 (tzw. "clipping")
    normalized_l_float = np.clip(normalized_l_float, 0, 255)
    
    # Konwertuj z powrotem do formatu 8-bitowego (uint8)
    final_l_channel = normalized_l_float.astype(np.uint8)
    
    print(f"INFO: Jasność znormalizowana. Średnia przed: {current_avg_l:.2f}, po: {np.mean(final_l_channel):.2f}")
    # --- KONIEC MODYFIKACJI ---

    # Krok 5: Połącz nowy, znormalizowany kanał L z oryginalnymi kanałami kolorów A i B
    final_lab_image = cv2.merge([final_l_channel, a_channel, b_channel])
    
    # Krok 6: Konwertuj obraz z powrotem do BGR
    final_bgr_image = cv2.cvtColor(final_lab_image, cv2.COLOR_LAB2BGR)
    
    return final_bgr_image

def main():
    """
    Główna funkcja skryptu do przetwarzania wsadowego.
    """
    if not os.path.isdir(INPUT_FOLDER):
        print(f"BŁĄD: Folder wejściowy '{INPUT_FOLDER}' nie istnieje!")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    files_to_process = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    
    if not files_to_process:
        print(f"W folderze '{INPUT_FOLDER}' nie znaleziono żadnych obsługiwanych plików.")
        return

    print(f"\nZnaleziono {len(files_to_process)} plików. Rozpoczynam przetwarzanie...")

    for filename in files_to_process:
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        # Używamy nowej funkcji
        processed_image = reduce_shadows_and_normalize(input_path)
        
        if processed_image is not None:
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}_znormalizowany.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            try:
                cv2.imwrite(output_path, processed_image)
                print(f"✅ ZAPISANO: {output_path}\n")
            except Exception as e:
                print(f"BŁĄD: Nie udało się zapisać pliku {output_path}. Powód: {e}\n")

    print("--- Zakończono pracę! ---")

if __name__ == '__main__':
    main()