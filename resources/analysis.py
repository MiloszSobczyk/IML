import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import shutil


# Funkcja do wczytania spektrogramu
def load_spectrogram(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Funkcja do przekształcenia intensywności pikseli na decybele
def convert_to_decibels(spect, min_db=-80):
    return (spect / 255.0) * (-min_db) + min_db


# Funkcja do przekształcenia wartości w dB na amplitudy
def read_amplitudes(spect_db):
    return 10 ** (spect_db / 20.0)


# Funkcja do analizy decybeli pikseli
def decibel_analysis(spect_db, folder_path, file_name, log_file):
    plt.figure(figsize=(10, 6))
    plt.hist(spect_db.ravel(), bins=256, color='black', alpha=0.7)
    plt.title("Histogram wartości w dB")
    plt.xlabel("Intensywność (dB)")
    plt.ylabel("Liczba pikseli")
    plt.savefig(os.path.join(folder_path, f"{file_name}_decibel_histogram"), bbox_inches='tight')
    plt.close()

    mean_db = np.mean(spect_db)
    median_db = np.median(spect_db)
    variance_db = np.var(spect_db)

    with open(log_file, 'a') as f:
        f.write(f"Średnia wartość w dB: {mean_db}\n")
        f.write(f"Mediana wartości w dB: {median_db}\n")
        f.write(f"Wariancja wartości w dB: {variance_db}\n")


# Funkcja do analizy intensywności pikseli
def pixel_intensity_analysis(img, folder_path, file_name, log_file):
    plt.figure(figsize=(10, 6))
    plt.hist(img.ravel(), bins=256, color='black', alpha=0.7)
    plt.title("Histogram intensywności pikseli")
    plt.xlabel("Intensywność")
    plt.ylabel("Liczba pikseli")
    plt.savefig(os.path.join(folder_path, f"{file_name}_pixel_intensity"), bbox_inches='tight')
    plt.close()

    mean_intensity = np.mean(img)
    median_intensity = np.median(img)
    variance_intensity = np.var(img)

    with open(log_file, 'a') as f:
        f.write(f"Średnia intensywność pikseli: {mean_intensity}\n")
        f.write(f"Mediana intensywności pikseli: {median_intensity}\n")
        f.write(f"Wariancja intensywności pikseli: {variance_intensity}\n")


# Funkcja do obliczenia SNR i zapisu do pliku
def calculate_snr(spect_db, log_file, description, samplerate=44100, binsize=512):
    amplitude_spect = read_amplitudes(spect_db)
    human_1 = 300
    human_2 = 3000
    freq_bins = amplitude_spect.shape[0]
    frequencies = np.fft.rfftfreq(binsize, d=1.0 / samplerate)[:freq_bins]
    bin_1 = np.argmax(frequencies > human_1)
    bin_2 = np.argmax(frequencies > human_2)
    E = amplitude_spect ** 2
    N = np.min(E[bin_1:bin_2]) * (bin_2 - bin_1 + 1) * E.shape[1]
    S = np.sum(E[bin_1:bin_2])
    snr = (S - N) / N
    snr_db = 10 * np.log10(snr)

    with open(log_file, 'a') as f:
        f.write(f"{description} SNR (w dB): {snr_db}\n")


# Funkcja do analizy rozkładu dB względem częstotliwości i czasu
def decibel_distribution_analysis(spect_db, folder_path, file_name):
    vertical_sum = np.mean(spect_db, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(vertical_sum)
    plt.title("Średnia intensywność w dB w zależności od częstotliwości")
    plt.xlabel("Częstotliwość (pasmo)")
    plt.ylabel("Średnia intensywność (dB)")
    plt.savefig(os.path.join(folder_path, f"{file_name}_decibel_vs_frequency"), bbox_inches='tight')
    plt.close()

    horizontal_sum = np.mean(spect_db, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(horizontal_sum)
    plt.title("Średnia intensywność w dB w zależności od czasu")
    plt.xlabel("Czas")
    plt.ylabel("Średnia intensywność (dB)")
    plt.savefig(os.path.join(folder_path, f"{file_name}_decibel_vs_time"), bbox_inches='tight')
    plt.close()


# Funkcja do wykrywania anomalii na podstawie wartości w dB
def anomaly_detection(spect_db, log_file):
    z_scores = stats.zscore(spect_db.ravel())
    anomalies = z_scores > 3
    with open(log_file, 'a') as f:
        f.write(f"Liczba wykrytych anomalii: {np.sum(anomalies)}\n")


# Funkcja do analizy spektrogramu z obliczeniem SNR dla obu wersji (z szumem i bez)
def analyze_spectrogram(image_path, noise_image_path):
    spectrogram = load_spectrogram(image_path)
    spect_db = convert_to_decibels(spectrogram)

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    new_folder_path = './analyze/' + file_name
    os.makedirs(new_folder_path, exist_ok=True)
    log_file = os.path.join(new_folder_path, f"{file_name}_analysis.txt")

    # Analiza intensywności pikseli
    pixel_intensity_analysis(spectrogram, new_folder_path, file_name, log_file)

    # Analiza intensywności w dB
    decibel_analysis(spect_db, new_folder_path, file_name, log_file)

    # Obliczenie SNR dla wersji bez szumu
    calculate_snr(spect_db, log_file, description="Bez szumu")

    # Obliczenie SNR dla wersji z szumem, jeśli plik istnieje
    if os.path.exists(noise_image_path):
        noise_spectrogram = load_spectrogram(noise_image_path)
        noise_spect_db = convert_to_decibels(noise_spectrogram)
        calculate_snr(noise_spect_db, log_file, description="Z szumem")

    # Analiza rozkładu dB względem częstotliwości i czasu
    decibel_distribution_analysis(spect_db, new_folder_path, file_name)

    # Wykrywanie anomalii w wartościach dB
    anomaly_detection(spect_db, log_file)


# Funkcja do przetworzenia wszystkich plików .png w folderze ./spectrograms/cleanraw/
def process_folder(folder_path='./spectrograms/spectrograms_enhanced/spectrograms_2/cleanraw'):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            noise_image_path = image_path.replace('spectrograms_enhanced/spectrograms_2', '').replace(
                '_spectrogram_', '_spectrogram_with_noise_')
            print(f"Analizowanie pliku: {image_path}")
            analyze_spectrogram(image_path, noise_image_path)


def reorganize_folders(base_dir):
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path) and "_" in item:
            parts = item.split('_')
            main_type = parts[0]
            script_type = parts[1]
            spectrogram_folder = '_'.join(parts[2:])

            main_type_dir = os.path.join(base_dir, main_type)
            os.makedirs(main_type_dir, exist_ok=True)

            script_dir = os.path.join(main_type_dir, script_type)
            os.makedirs(script_dir, exist_ok=True)

            new_path = os.path.join(script_dir, spectrogram_folder)
            shutil.move(item_path, new_path)
            print(f"Moved {item} to {new_path}")

