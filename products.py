class Products:
    def __init__(self, tag, bot_response):
        self.tag = tag
        self.bot_response = bot_response

    def recommend_laptop(self):
        if self.tag == "laptop":
            # Burada laptop için önerilerinizi gerçekleştirebilirsiniz.
            return self.bot_response + "DENEMEEEEEE"
        else:
            return "Bu ürün için öneri bulunmamaktadır."

    def get_user_input():
        ram = input("Lütfen Ram miktarını giriniz: ")
        screen_size = input("Lütfen Ekran Boyutunu giriniz(inch): ")
        color = input("Lütfen Renk seçiniz: ")
        brand = input("Lütfen Marka seçiniz: ")
        price = input("Lütfen Fiyat aralığını giriniz: ")

        return ram, screen_size, color, brand, price

    # Dispatch tablosu
    dispatch_table = {
        "laptop": recommend_laptop,
    }