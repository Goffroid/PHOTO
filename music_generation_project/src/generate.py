import torch                            
import numpy as np                      
from model import MusicLSTM             
import os                               

# Загрузка обученной модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Определение устройства
print(f"Используется устройство: {device}") # Вывод информации об устройстве

# Определение размеров модели на основе данных
data_path = "data/processed/notes.npy" # Путь к обработанным данным
if not os.path.exists(data_path):      # Проверка существования файла
    raise FileNotFoundError(f"Файл {data_path} не найден.")

data = np.load(data_path)              # Загрузка обработанных данных
print(f"Форма загруженных данных: {data.shape}") # Вывод формы данных

if len(data.shape) == 1:               # Если данные одномерные
    input_size = 1                      # Размер входа 1
else:                                  # Если данные многомерные
    input_size = data.shape[1]         # Размер входа по второму измерению

hidden_size = 128                       # Размер скрытого состояния
output_size = input_size                # Размер выхода равен размеру входа

# Создание и загрузка модели
model = MusicLSTM(input_size, hidden_size, output_size) # Создание экземпляра
model_path = "models/music_lstm.pth"   # Путь к модели

if not os.path.exists(model_path):     # Проверка существования модели
    raise FileNotFoundError(f"Модель {model_path} не найдена. Сначала запустите train.py")

model.load_state_dict(torch.load(model_path, map_location=device)) # Загрузка весов
model.to(device)                        # Перемещение модели на устройство
model.eval()                            # Режим оценки
print("Модель успешно загружена")      # Сообщение об успешной загрузке

# Генерация последовательности
seed_length = 50                        # Длина начальной последовательности
if len(data) < seed_length:             # Проверка достаточности данных
    seed_length = len(data)             # Корректировка длины

# Использование реальных данных как seed
seed_sequence = data[:seed_length]      # Начальная последовательность
if len(seed_sequence.shape) == 1:       # Если данные одномерные
    seed_sequence = seed_sequence.reshape(1, -1, 1) # Преобразование в 3D
else:                                  # Если данные многомерные
    seed_sequence = seed_sequence.reshape(1, seed_length, input_size) # Преобразование в 3D

seed_tensor = torch.FloatTensor(seed_sequence).to(device) # Преобразование в тензор
print(f"Форма seed_tensor: {seed_tensor.shape}") # Вывод формы

generated = []                          # Список для сгенерированных нот
num_notes_to_generate = 100             # Количество нот для генерации

print(f"\nГенерация {num_notes_to_generate} нот...") # Сообщение о начале генерации

with torch.no_grad():                   # Отключение вычисления градиентов
    current_sequence = seed_tensor      # Начальная последовательность
    
    for i in range(num_notes_to_generate): # Цикл генерации
        output = model(current_sequence) # Получение предсказания
        generated_note = output.cpu().numpy().flatten() # Преобразование в numpy
        generated.append(generated_note) # Добавление ноты в список
        
        # Обновление последовательности
        new_note = output.unsqueeze(1)  # Добавление размерности
        current_sequence = torch.cat([current_sequence[:, 1:, :], new_note], dim=1)
        
        if (i + 1) % 20 == 0:           # Вывод прогресса каждые 20 нот
            print(f"Сгенерировано {i + 1}/{num_notes_to_generate} нот")

# Сохранение результата
generated_array = np.array(generated)   # Преобразование в numpy массив
save_path = "generated_music.npy"       # Путь для сохранения
np.save(save_path, generated_array)     # Сохранение как numpy-массив
print(f"\nСгенерировано нот: {len(generated)}") # Вывод количества нот
print(f"Форма сгенерированных данных: {generated_array.shape}") # Вывод формы
print(f"Результат сохранён в {save_path}") # Сообщение о сохранении

# Базовая статистика сгенерированных данных
print(f"\nСтатистика сгенерированных нот:") # Заголовок статистики
print(f"Диапазон значений: {generated_array.min():.4f} - {generated_array.max():.4f}") # Диапазон
print(f"Среднее значение: {generated_array.mean():.4f}") # Среднее значение
print(f"Стандартное отклонение: {generated_array.std():.4f}") # Стандартное отклонение