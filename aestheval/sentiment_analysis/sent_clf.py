from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification, set_seed
import re

def _remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def _remove_escaped_characters(text: str):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    # When the scaped character is just before a ".", removing the escaped char will remove the
    # space between the dot and the new word. This is a quick workaround.
    return text.translate(translator).replace(".", ". ").replace(".  ", ". ")

def _clean_text(text, remove_url=True, remove_escaped_chars=True):
    if remove_url:
        text=_remove_URL(text)
    if remove_escaped_chars:
        text=_remove_escaped_characters(text)
    return text


class Predictor():

    def __init__(self, model_path: str) -> None:
        
        set_seed(1997) #Lucky seed
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        #Uses the gpu 0 by default
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True, device=0)

    def predict(self, texts:  list['str']):
        cleaned_texts = [_clean_text(text) for text in texts]
        results = self.pipe(cleaned_texts, truncation=True, max_length=512)
        # Reorder result
        results = [{label['label']: label['score']for label in result} for result in results]
        return results