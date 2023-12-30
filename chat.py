import random
import json
import torch

from products import Products
# Fonksiyonlar ve model
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# intents.json dosyası okunarak içeriği intents adlı değişkende saklanır
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Eğitilmiş modelin ağırlıkları ve gerekli diğer bilgiler yüklenir
FILE = "data.pth"
data = torch.load(FILE)

# Model için gerekli boyut bilgileri ve sözlükler belirlenir
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Model mimarisi tanımlanır ve eğitilmiş ağırlıklar yüklenir
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Chat botunun adı
bot_name = "Sam"

# Kullanıcının girişine göre bir yanıt almak için bir fonksiyon tanımlanır
def get_response(msg):
    # Kullanıcının girişi tokenize edilir ve bag of words'e dönüştürülür
    print("-----------------------------------------------------------")
    sentence = tokenize(msg)
    print(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Model üzerinden tahmin yapılır ve en yüksek olasılıklı etiket seçilir
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Eğer tahminin olasılığı belirli bir eşiği geçiyorsa, uygun yanıt seçilir
    tag = tags[predicted.item()]
    print(tag)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "laptop":
                    return "DENEME"
                return random.choice(intent['responses'])
    
    # Eğer tahmin belirlenen eşik değerinin altındaysa anlaşılmadı mesajı döndürülür
    return "Anlayamadım..."

# Ana döngü, kullanıcının çeşitli girişlerine yanıt almak için kullanılır
if __name__ == "__main__":
    print("Hadi konuşalım! (çıkmak için 'quit' yazın)")
    while True:
        sentence = input("Ben: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
