# Импорт библиотек
import pretty_midi                     # Для работы с MIDI-файлами
import numpy as np                     # Для численных операций
import os                              # Для работы с файловой системой
from tqdm import tqdm                  # Для прогресс-бара

# Функция для загрузки и обработки MIDI-файлов
def load_midi_files(directory):
    notes = []                         # Список для хранения нот
    midi_files = []                    # Список для хранения путей к файлам
    
    # Поиск MIDI-файлов
    for root, dirs, files in os.walk(directory): # Рекурсивный обход директории
        for file in files:              # Перебор файлов
            if file.endswith('.midi') or file.endswith('.mid'): # Проверка формата
                midi_files.append(os.path.join(root, file)) # Добавление пути
    
    print(f"Найдено MIDI-файлов: {len(midi_files)}") # Вывод количества файлов
    
    # Обработка файлов с прогресс-баром
    for file_path in tqdm(midi_files[:100], desc="Обработка MIDI-файлов"): # Ограничение 100 файлов
        try:                            # Обработка исключений
            midi = pretty_midi.PrettyMIDI(file_path) # Загрузка MIDI-файла
            for instrument in midi.instruments: # Перебор инструментов
                for note in instrument.notes:   # Перебор нот
                    # Сохранение pitch, start, end, velocity
                    notes.append([note.pitch, note.start, note.end, note.velocity])
        except Exception as e:          # Обработка ошибок
            print(f"Ошибка при обработке {file_path}: {e}") # Вывод ошибки
    
    return np.array(notes, dtype=np.float32) # Возврат массива нот

# Основная логика
if __name__ == "__main__":
    raw_data_path = "data/raw/maestro-v3.0.0-midi" # Путь к исходным данным
    
    # Проверка существования директории
    if not os.path.exists(raw_data_path): # Проверка пути
        print(f"Директория {raw_data_path} не найдена.") # Сообщение об ошибке
        print("Распакуйте maestro-v3.0.0-midi.zip в data/raw/") # Инструкция
    else:                               # Если директория существует
        processed_data = load_midi_files(raw_data_path) # Обработка данных
        print(f"Обработано нот: {len(processed_data)}") # Вывод количества нот
        print(f"Форма данных: {processed_data.shape}") # Вывод формы данных
        
        # Создание директории для обработанных данных
        os.makedirs("data/processed", exist_ok=True) # Создание папки
        
        # Сохранение данных
        save_path = "data/processed/notes.npy" # Путь для сохранения
        np.save(save_path, processed_data) # Сохранение данных
        print(f"Данные сохранены в {save_path}") # Сообщение о сохранении
        
        # Базовая статистика
        if len(processed_data) > 0:     # Если есть данные
            print(f"Диапазон pitch: {processed_data[:, 0].min()} - {processed_data[:, 0].max()}") # Статистика pitch
            print(f"Диапазон start: {processed_data[:, 1].min():.2f} - {processed_data[:, 1].max():.2f}") # Статистика start