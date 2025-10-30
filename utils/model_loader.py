
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np
from utils.fusion_utils import *

class EmbeddingModel:

    #all-MiniLM-L6-v2
    def __init__(self, model_name='sentence-transformers/bert-base-nli-mean-tokens', device=None, nlp=None):
        if device is None:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
        else:
            if 'cuda' in device and not torch.cuda.is_available():
                print(f"Warning: CUDA is not available, falling back to CPU.")
                device = 'cpu'
            else:
                if 'cuda' in device:
                    gpu_index = int(device.split(':')[-1])
                    if gpu_index >= torch.cuda.device_count():
                        print(f"Warning: GPU index {gpu_index} out of range, falling back to CPU.")
                        device = 'cpu'
        
        self.device = device
        self.dim = 768
        self.model_name = model_name

        print(f"[EmbeddingModel] Using device: {self.device}")
        print(f"[Model] Using model: {self.model_name}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        
        # 使用spaCy模型进行预处理
        #self.nlp = nlp or spacy.load("en_core_web_sm")

        self.default_embedding = np.zeros((768,))

    @staticmethod
    def _pre(text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return text
                 
    def generate_embedding(self, texts, batch_size: int = 128) -> np.ndarray:
        if isinstance(texts, str) or is_null(texts):                          # 允许传单条
            texts = [texts]

        n = len(texts)
        embs_np = np.zeros((n, self.dim), dtype="float32")  # 先全部置零

        # ------- 1) 筛出非空文本 -------
        non_empty_idx, non_empty_texts = [], []
        for i, t in enumerate(texts):
            if t is None or str(t).strip() == "" or str(t).lower() == "null":
                continue
            non_empty_idx.append(i)
            non_empty_texts.append(self._pre(str(t)))

        if not non_empty_texts:             # 全为空，直接返回零矩阵
            return embs_np

        # ------- 2) 分批 tokenizer → model -------
        self.model.eval()
        with torch.no_grad():
            for b in range(0, len(non_empty_texts), batch_size):
                sub_texts = non_empty_texts[b : b + batch_size]

                batch = self.tokenizer(
                    sub_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**batch)
                vec = outputs.last_hidden_state[:, 0, :]     # [CLS]
                vec = F.normalize(vec, p=2, dim=1)           # L2 归一化
                vec = vec.cpu().numpy().astype("float32")

                # ------- 3) 写回对应行 -------
                start = b
                for offset, row in enumerate(vec):
                    embs_np[non_empty_idx[start + offset]] = row

        return embs_np
    
    __call__ = generate_embedding

