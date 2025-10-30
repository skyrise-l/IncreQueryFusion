from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatBot:
    def __init__(self, model_name="/home/lwh/model/DeepSeek-R1-Distill-Llama-8B", device: str = "cuda:5"):
        # 下载并加载模型和分词器
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 确保 CUDA 设备可用
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        #self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def talk(self, input_text, max_length: int = 4096, num_return_sequences: int = 1):
        # 使用分词器进行编码
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            

            # 使用模型进行推理 
            outputs = self.model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],  
                max_length=max_length, 
                num_return_sequences=num_return_sequences
            )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
        except Exception as e:
            print(e)
            return f"llm server error {e}"