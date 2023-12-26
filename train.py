import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Özel fonksiyonlar ve sınıflar import edilir
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# intents.json dosyası okunur
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Kullanılacak listeler tanımlanır
all_words = []
tags = []
xy = []

# Her bir niyetin desenlerinde dolaşılır
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # Etiket listesine eklenir
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Cümle parçalanır (tokenize edilir)
        all_words.extend(w)  # Tüm kelimeler listesine eklenir
        xy.append((w, tag))  # Kelime ve etiket ikilisi xy listesine eklenir

# Bazı özel karakterler hariç tüm kelimeler köklenir
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # Tekrar eden kelimeler kaldırılır ve sıralanır
tags = sorted(set(tags))  # Tekrar eden etiketler kaldırılır ve sıralanır

print(len(xy), "desen")
print(len(tags), "etiket:", tags)
print(len(all_words), "benzersiz köklü kelime:", all_words)

# Eğitim verisi oluşturulur
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Her desen için bag of words vektörü oluşturulur
    X_train.append(bag)
    label = tags.index(tag)  # Etiketlerin indeksi alınır
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parametreler tanımlanır
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(input_size, output_size)

# Özel veri seti sınıfı tanımlanır
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Veri seti oluşturulur ve yüklenir
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model tanımlanır ve cihaza yüklenir
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Kayıp fonksiyonu ve optimize edici tanımlanır
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model eğitilir
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Kayıp: {loss.item():.4f}')

print(f'final kayıp: {loss.item():.4f}')

# Eğitilen modelin durumu ve diğer bilgiler kaydedilir
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'eğitim tamamlandı. dosya {FILE} olarak kaydedildi')
