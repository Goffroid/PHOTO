import numpy as np                      
import pretty_midi                      
import os                               
import matplotlib.pyplot as plt         

# Функция для преобразования данных в MIDI
def npy_to_midi(npy_path, output_midi="generated_music.mid", note_duration=0.5, velocity=100):
    """
    Конвертирует файл .npy в MIDI файл
    npy_path: путь к файлу .npy
    output_midi: выходной MIDI файл
    note_duration: длительность нот в секундах
    velocity: громкость нот (0-127)
    """
    # Загрузка сгенерированных данных
    data = np.load(npy_path)            # Загрузка данных из файла
    print(f"Загруженные данные: {data.shape}") # Вывод формы данных
    
    # Создание объекта MIDI
    midi = pretty_midi.PrettyMIDI()     # Создание пустого MIDI файла
    instrument = pretty_midi.Instrument(program=0) # Создание инструмента (пианино)
    
    # Определение формата данных
    if len(data.shape) == 1:            # Если одномерный массив (только pitch)
        pitches = data                  # Использование напрямую
    elif data.shape[1] == 1:            # Если двумерный с одним столбцом
        pitches = data.flatten()        # Преобразование в одномерный
    else:                               # Если несколько столбцов
        pitches = data[:, 0]            # Использование первого столбца (pitch)
    
    # Преобразование pitch в целые числа и ограничение диапазона
    pitches = np.round(pitches).astype(int) # Округление до целых
    pitches = np.clip(pitches, 21, 108)     # Ограничение диапазона пианино (21-108)
    
    print(f"Диапазон pitch: {pitches.min()} - {pitches.max()}") # Вывод диапазона
    print(f"Уникальные ноты: {np.unique(pitches)}") # Вывод уникальных нот
    
    # Создание нот
    for i, pitch in enumerate(pitches): # Перебор всех нот
        start_time = i * note_duration   # Время начала ноты
        end_time = start_time + note_duration # Время окончания ноты
        note = pretty_midi.Note(        # Создание объекта ноты
            velocity=velocity,          # Громкость ноты
            pitch=pitch,                # Высота тона
            start=start_time,           # Время начала
            end=end_time                # Время окончания
        )
        instrument.notes.append(note)   # Добавление ноты в инструмент
    
    midi.instruments.append(instrument) # Добавление инструмента в MIDI
    midi.write(output_midi)             # Запись MIDI файла
    print(f"MIDI файл сохранён: {output_midi}") # Сообщение о сохранении
    print(f"Количество созданных нот: {len(pitches)}") # Вывод количества нот
    
    return midi, pitches                 # Возврат объектов для анализа

# Функция для визуализации музыки
def visualize_music(pitches, note_duration=0.5):
    """
    Визуализирует сгенерированную музыку
    pitches: массив высот нот
    note_duration: длительность каждой ноты
    """
    # Создание фигуры с несколькими графиками
    fig, axes = plt.subplots(3, 1, figsize=(12, 8)) # Создание 3 графиков
    
    # 1. График последовательности нот
    axes[0].plot(pitches, 'b-', linewidth=2, marker='o', markersize=4) # Линейный график
    axes[0].set_title('Последовательность нот (Pitch)') # Заголовок
    axes[0].set_xlabel('Номер ноты')                   # Подпись оси X
    axes[0].set_ylabel('Высота тона')                  # Подпись оси Y
    axes[0].grid(True, alpha=0.3)                     # Включение сетки
    
    # 2. Гистограмма распределения нот
    axes[1].hist(pitches, bins=range(20, 110, 5), alpha=0.7, color='green', edgecolor='black') # Гистограмма
    axes[1].set_title('Распределение нот по высоте')   # Заголовок
    axes[1].set_xlabel('Высота тона')                 # Подпись оси X
    axes[1].set_ylabel('Количество')                  # Подпись оси Y
    axes[1].grid(True, alpha=0.3)                     # Включение сетки
    
    # 3. Пианино-ролл представление
    for i, pitch in enumerate(pitches):               # Перебор нот
        rect = plt.Rectangle((i*note_duration, pitch-0.4), note_duration, 0.8, 
                           alpha=0.5, color='red')    # Создание прямоугольника
        axes[2].add_patch(rect)                       # Добавление на график
    
    axes[2].set_xlim(0, len(pitches)*note_duration)   # Ограничение по X
    axes[2].set_ylim(20, 110)                         # Ограничение по Y (диапазон пианино)
    axes[2].set_title('Пианино-ролл представление')   # Заголовок
    axes[2].set_xlabel('Время (секунды)')             # Подпись оси X
    axes[2].set_ylabel('Высота тона')                 # Подпись оси Y
    axes[2].grid(True, alpha=0.3)                     # Включение сетки
    
    plt.tight_layout()                                # Автоматическая настройка отступов
    plt.savefig('generated_music_visualization.png', dpi=150) # Сохранение графика
    plt.show()                                        # Отображение графиков
    
    print(f"Визуализация сохранена: generated_music_visualization.png") 

# Основная функция
if __name__ == "__main__":
    # Конвертация в MIDI
    npy_file = "generated_music.npy"    # Путь к сгенерированному файлу
    if os.path.exists(npy_file):        # Проверка существования файла
        midi, pitches = npy_to_midi(npy_file, "generated_music.mid") # Конвертация
        
        # Визуализация
        visualize_music(pitches)        # Создание графиков
        
        # Статистика
        print("\n=== СТАТИСТИКА СГЕНЕРИРОВАННОЙ МУЗЫКИ ===") # Заголовок
        print(f"Общее количество нот: {len(pitches)}") # Количество нот
        print(f"Длительность: {len(pitches) * 0.5:.1f} секунд") # Длительность
        print(f"Средняя высота тона: {pitches.mean():.1f}") # Средний pitch
        print(f"Стандартное отклонение: {pitches.std():.1f}") # Разброс
        
        # Названия нот (пример)
        note_names = {                  # Словарь для названий нот
            60: "До (C4)", 62: "Ре (D4)", 64: "Ми (E4)",
            65: "Фа (F4)", 67: "Соль (G4)", 69: "Ля (A4)", 71: "Си (B4)"
        }
        
        # Поиск часто встречающихся нот
        unique, counts = np.unique(pitches, return_counts=True) # Уникальные ноты и их количество
        top_notes = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5] # Топ-5 нот
        
        print("\nСамые частые ноты:")   # Заголовок
        for note, count in top_notes:  # Перебор топ-нот
            name = note_names.get(note, f"Нота {note}") # Получение названия
            print(f"  {name}: {count} раз ({count/len(pitches)*100:.1f}%)") # Вывод
    else:                               
        print(f"Файл {npy_file} не найден. Сначала запустите generate.py") 