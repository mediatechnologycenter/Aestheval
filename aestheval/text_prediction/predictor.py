from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification, set_seed
from aestheval.data.datautils.data_cleaning import clean_text


class Predictor():

    def __init__(self, model_path: str) -> None:
        
        set_seed(1997) #Lucky seed
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        #Uses the gpu 0 by default
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True, device=0)

    def predict(self, texts:  "list['str']"):
        cleaned_texts = [clean_text(text) for text in texts]
        results = self.pipe(cleaned_texts, truncation=True, max_length=512)
        # Reorder result
        results = [{label['label']: label['score']for label in result} for result in results]
        return results