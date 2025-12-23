import torch                            
import torch.nn as nn                   
import torch.optim as optim             
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np                      
from model import MusicLSTM             
import os                               

# Настройки для ускорения обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Использование GPU
print(f"Используется устройство: {device}") # Вывод информации об устройстве
torch.backends.cudnn.benchmark = True  # Оптимизация для NVIDIA GPU

# Загрузка данных с проверкой
data_path = "data/processed/notes.npy" # Путь к обработанным данным
if not os.path.exists(data_path):      # Проверка существования файла
    raise FileNotFoundError(f"Файл {data_path} не найден. Сначала запустите data_preprocessing.py")

data = np.load(data_path)              # Загрузка обработанных данных
print(f"Форма загруженных данных: {data.shape}") # Вывод формы данных
print(f"Тип данных: {data.dtype}")     # Вывод типа данных

# Проверка и подготовка данных
if len(data.shape) == 1:               # Если данные одномерные (только pitch)
    print("Обнаружены одномерные данные (только pitch)")
    data = data.reshape(-1, 1)         # Преобразование в 2D массив (N, 1)
    input_size = 1                      # Размер входа для модели
elif data.shape[1] >= 3:               # Если данные содержат pitch, start, end
    print("Обнаружены многомерные данные (pitch, start, end)")
    input_size = data.shape[1]         # Размер входа равен числу признаков
else:                                  # Другие случаи
    input_size = data.shape[1]         # Размер входа по второму измерению
    print(f"Размер входа установлен в: {input_size}")

# Подготовка последовательностей для обучения
seq_length = 50                        # Длина последовательности для обучения
total_sequences = len(data) - seq_length - 1 # Общее количество последовательностей
print(f"Всего последовательностей для обучения: {total_sequences}")

if total_sequences < 100:              # Проверка достаточности данных
    raise ValueError(f"Недостаточно данных. Нужно минимум 100 последовательностей, а есть {total_sequences}")

# Создание последовательностей и целей
sequences = []                          # Список для последовательностей
targets = []                            # Список для целей

for i in range(total_sequences):        # Перебор индексов
    seq = data[i:i+seq_length]          # Входная последовательность
    target = data[i+seq_length]         # Целевое значение
    sequences.append(seq)               # Добавление последовательности
    targets.append(target)              # Добавление цели

sequences = np.array(sequences)         # Преобразование в numpy массив
targets = np.array(targets)             # Преобразование в numpy массив

print(f"Форма sequences: {sequences.shape}") # Вывод формы sequences
print(f"Форма targets: {targets.shape}")     # Вывод формы targets

# Преобразование в тензоры PyTorch
seq_tensor = torch.FloatTensor(sequences).to(device) # Тензор последовательностей
target_tensor = torch.FloatTensor(targets).to(device) # Тензор целей

print(f"Форма seq_tensor: {seq_tensor.shape}") # Вывод формы seq_tensor
print(f"Форма target_tensor: {target_tensor.shape}") # Вывод формы target_tensor

# Создание DataLoader
dataset = TensorDataset(seq_tensor, target_tensor) # Создание набора данных
batch_size = 32                          # Размер батча
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Загрузчик с батчами
print(f"Количество батчей: {len(dataloader)}") # Вывод количества батчей

# Инициализация модели, оптимизатора и функции потерь
hidden_size = 128                         # Размер скрытого состояния
output_size = input_size                  # Размер выхода равен размеру входа
model = MusicLSTM(input_size, hidden_size, output_size).to(device) # Модель
optimizer = optim.Adam(model.parameters(), lr=0.001) # Оптимизатор Adam
criterion = nn.MSELoss()                  # Функция потерь для регрессии

print(f"Архитектура модели:")            # Вывод информации о модели
print(model)                             # Вывод архитектуры модели

# Цикл обучения
num_epochs = 10                          # Количество эпох
print(f"\nНачало обучения на {num_epochs} эпох...") # Сообщение о начале обучения

for epoch in range(num_epochs):          # Цикл по эпохам
    epoch_loss = 0.0                     # Сумма потерь за эпоху
    num_batches = 0                      # Счетчик батчей
    
    for batch_seqs, batch_targets in dataloader: # Итерация по батчам
        optimizer.zero_grad()             # Обнуление градиентов
        output = model(batch_seqs)        # Прямой проход
        loss = criterion(output, batch_targets) # Расчёт потерь
        loss.backward()                   # Обратное распространение
        optimizer.step()                  # Обновление весов
        
        epoch_loss += loss.item()         # Добавление потерь
        num_batches += 1                  # Увеличение счетчика
    
    avg_loss = epoch_loss / num_batches   # Средние потери за эпоху
    print(f"Эпоха {epoch+1}/{num_epochs}, Средние потери: {avg_loss:.6f}") # Вывод потерь

# Сохранение модели
model_save_path = "models/music_lstm.pth" # Путь для сохранения модели
os.makedirs("models", exist_ok=True)      # Создание папки models если её нет
torch.save(model.state_dict(), model_save_path) # Сохранение весов модели
print(f"\nМодель сохранена в {model_save_path}") 