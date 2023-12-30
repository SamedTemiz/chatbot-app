class Chatbox {
    constructor() {
        // Gerekli DOM elementleri tanımlanır
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        // Başlangıçta chatbox'ın durumu (açık/kapalı) ve mesajlar dizisi tanımlanır
        this.state = false;
        this.messages = [];
    }

    // Chatbox'ı görüntülemek için gereken event listener'lar tanımlanır
    display() {
        const {openButton, chatBox, sendButton} = this.args;

        // Chatbox'ın açılıp kapanması için event listener eklenir
        openButton.addEventListener('click', () => this.toggleState(chatBox))

        // Mesaj gönderme butonuna event listener eklenir
        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        // Klavyeden Enter tuşuna basıldığında da mesaj gönderme fonksiyonu çalıştırılır
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    // Chatbox'ın durumunu değiştirir (açık <-> kapalı)
    toggleState(chatbox) {
        this.state = !this.state;

        // Chatbox'ın görünürlüğünü ayarlar
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    // Mesaj gönderme
    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        // Kullanıcı mesajı oluşturulur ve mesajlar dizisine eklenir
        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        // Sunucuya mesaj gönderilir ve yanıt beklenir
        //'http://127.0.0.1:5000/predict'
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            // Sunucudan gelen yanıt alınır ve mesajlar dizisine eklenir
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            // Hata durumunda konsola hata yazdırılır
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    // Mesajların görüntülendiği alan güncellenir
    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            // Sam (chatbot) veya kullanıcıya göre mesajlar oluşturulur
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        // Güncellenen mesajlar chatbox içerisine eklenir
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

// Chatbox sınıfından bir örnek oluşturulur ve görüntüleme fonksiyonu çağrılır
const chatbox = new Chatbox();
chatbox.display();
