from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

class Model:
    def __init__(self, model, device="cuda"):
        self.model_name = model
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model)
        self.model.to(self.device)
    
    def evaluate(self, batch):
        inputs = self.processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    