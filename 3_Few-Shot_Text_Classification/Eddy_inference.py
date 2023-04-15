"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro  
"""   
from setfit import SetFitModel

model = SetFitModel.from_pretrained("./output")

# Realizamos a Inferência
#preds = model(["adorei o filme do homem aranha!", "abacaxi na pizza é o pior 🤮"])
#preds = model(["você está horrível", "ótima performance", "pessoas gordas me enojam"])
preds = model(["adorei o filme do homem aranha!",
    "abacaxi na pizza é o pior 🤮",
    "que porra é essa peça",
    "bom dia, senhora chefe",
    "o produto é excelente",
    "um pedaço de lixo"])
print(preds)


import numpy as np 
import pandas as pd
results = list(np.array(preds))

def return_label(val, labels=['Positivo', 'Negativo']):
    if val == 1:
        result = labels[0]
    else:
        result = labels[1]
    return result

results = map(return_label, results)
sentimento = list(results)
print(sentimento)
