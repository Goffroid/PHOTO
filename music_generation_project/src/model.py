# Импорт библиотек
import torch                            # Основная библиотека PyTorch
import torch.nn as nn                   # Модуль для нейронных сетей
import torch.nn.functional as F         # Функциональные модули

# Определение модели LSTM
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicLSTM, self).__init__() # Инициализация родительского класса
        self.hidden_size = hidden_size   # Размер скрытого состояния
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True) # Первый LSTM-слой
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True) # Второй LSTM-слой
        self.dropout = nn.Dropout(0.2)   # Слой dropout для регуляризации
        self.fc = nn.Linear(hidden_size, output_size) # Полносвязный слой
    
    def forward(self, x):
        out, _ = self.lstm1(x)           # Прямой проход через первый LSTM
        out, _ = self.lstm2(out)         # Прямой проход через второй LSTM
        out = self.dropout(out)          # Применение dropout
        out = self.fc(out[:, -1, :])     # Использование последнего выхода
        return out                       # Возврат выходного тензора

# Создание экземпляра модели для тестирования
if __name__ == "__main__":
    model = MusicLSTM(input_size=3, hidden_size=128, output_size=3) # Инициализация
    print("Архитектура модели:")        # Вывод заголовка
    print(model)                        # Вывод архитектуры модели
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters())}") # Вывод числа параметров