class Products:
    def __init__(self, tag, response):
        self.tag = tag
        self.response = response

    def recommend_laptop(self):
        if self.tag == "laptop":
            # Burada laptop için önerilerinizi gerçekleştirebilirsiniz.
            return "BURASI ÇALIŞTI"
        else:
            return "Bu ürün için öneri bulunmamaktadır."
