"""
Engenheiro de AI Sênior --> Ng Wai Foong
"""
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset

# Carregamos nossos Datsets
dataset = load_dataset('csv', data_files={
    'train': ['./data/train.csv'],
    'eval': ['./data/eval.csv']},
    #cache_dir="./data/"
)
print(dataset['train'])
print(dataset['eval'])


# Carregamos um Modelo SetFit do Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cache_dir="/home/eddygiusepe/1_Eddy_Giusepe/4_SetFit_Few-Shot_Learning/SetFit_Efficient_Few-Shot_Learning_without_Prompts/3_Few-Shot_Text_Classification/models"
)
# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=1,
    num_iterations=2,  # O número de pares de texto a serem gerados para APRENDIZADO CONTRASTIVO
    num_epochs=1,  # O número de épocas a serem usadas para APRENDIZADO CONTRASTIVO
    column_mapping={"text": "text", "label": "label"}  # Mapeamos as colunas do dataset text/label esperado pelo trainer
)

# Treinamos e avaliamos
trainer.train()
metrics = trainer.evaluate()

# save
trainer.model._save_pretrained(save_directory="/home/eddygiusepe/1_Eddy_Giusepe/4_SetFit_Few-Shot_Learning/SetFit_Efficient_Few-Shot_Learning_without_Prompts/3_Few-Shot_Text_Classification/output/")
