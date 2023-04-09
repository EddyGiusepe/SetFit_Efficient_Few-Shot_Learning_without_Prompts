"""
Engenheiro de AI Sênior --> Ng Wai Foong
"""
from setfit import SetFitModel

model = SetFitModel.from_pretrained("./output/", local_files_only=True)

sentiment_dict = {"negative": "0", "positive": "1"}
inverse_dict = {value: key for (key, value) in sentiment_dict.items()}

# Run inference
text_list = [
    "adorei o filme do homem aranha!",
    "abacaxi na pizza é o pior",
    "que porra é essa peça",
    "bom dia, senhora chefe",
    "o produto é excelente",
    "um pedaço de lixo"
]

preds = model(text_list)

for i in range(len(text_list)):
    print(text_list[i])
    print(inverse_dict[preds[i]])
    print('\n')
    