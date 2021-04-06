#!/usr/bin/env python
# coding: utf-8

# # **Fine-tuning XLSR-Wav2Vec2 for Multi-Lingual ASR with 🤗 Transformers**

# ## Pre-configuration

# In[1]:


from ipywidgets import widgets


# In[2]:


import os


# In[3]:


language_code = 'ga-IE'
language_name = 'irish'
base_model = "facebook/wav2vec2-large-xlsr-53"
pretrain_model = f"jimregan/wav2vec2-large-xlsr-{language_name}-extra6"

data_dir = f"/workspace/data/{language_code}"
output_models_dir = f"/workspace/output_models/{language_code}/wav2vec2-large-xlsr-{language_name}-extra6"


# In[20]:


from datasets import load_dataset, load_metric, Dataset

merged_train = Dataset.load_from_disk('/workspace/data/irish/preprocessed/train')
merged_valid = Dataset.load_from_disk('/workspace/data/irish/preprocessed/test')
common_voice_test = load_dataset("common_voice", language_code, split="test")


# In[5]:


common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])


# In[6]:


def is_upper_vowel(letter):
  if letter in ['A', 'E', 'I', 'O', 'U', 'Á', 'É', 'Í', 'Ó', 'Ú']:
    return True
  else:
    return False

def irish_lower(word):
  if len(word) > 1 and word[0] in ['n', 't'] and is_upper_vowel(word[1]):
    return word[0] + '-' + word[1:].lower()
  else:
    return word.lower()

def irish_lower_sentence(sentence):
  return " ".join([irish_lower(w) for w in sentence.split(" ")])


# In[7]:


import re
chars_to_ignore_regex = '[,\?\.\!\;\:\"\“\%\‘\”\(\)\*\–]'

def remove_special_characters(batch):
    tmp = re.sub('’ ', ' ', batch["sentence"])
    tmp = re.sub("’$", '', tmp)
    tmp = re.sub('’', '\'', tmp)
    tmp = re.sub(chars_to_ignore_regex, '', tmp)
    batch["sentence"] = irish_lower_sentence(tmp).strip() + ' '
    return batch


# In[8]:


common_voice_test = common_voice_test.map(remove_special_characters)


# In[9]:


vocab_list = [char for char in "aábcdeéfghiíjklmnoópqrstuúvwxyz'- "]
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict


# In[10]:


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


# In[11]:


vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)


# In[12]:


import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)


# In[13]:


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


# In[14]:


from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


# In[15]:


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# In[16]:


processor.save_pretrained(output_models_dir)


# In[17]:


import torchaudio

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


# In[23]:


common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)


# In[24]:


import librosa
import numpy as np

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), batch["sampling_rate"], 16_000)
    batch["sampling_rate"] = 16_000
    return batch


# In[25]:


common_voice_test = common_voice_test.map(resample, num_proc=12)
merged_train = merged_train.map(resample, num_proc=12)
merged_valid = merged_valid.map(resample, num_proc=12)


# In[26]:


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


# In[27]:


merged_train = merged_train.map(prepare_dataset, remove_columns=merged_train.column_names, batch_size=8, num_proc=12, batched=True)
merged_valid = merged_valid.map(prepare_dataset, remove_columns=merged_valid.column_names, batch_size=8, num_proc=12, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=12, batched=True)


# In[28]:


import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# In[29]:


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# In[30]:


wer_metric = load_metric("wer")


# In[31]:


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# In[32]:


from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.4,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)


# In[33]:


model.freeze_feature_extractor()


# In[34]:


from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=output_models_dir,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=60,
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=20,
)


# In[35]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=merged_train,
    eval_dataset=merged_valid,
    tokenizer=processor.feature_extractor,
)


# In[36]:


trainer.train()


























# In[ ]:


trainer.save_model(output_models_dir)
tokenizer.save_pretrained(output_models_dir)


# In[ ]:


trainer.save_model('/workspace/output_models/newest-run')
tokenizer.save_pretrained('/workspace/output_models/newest-run')


# In[ ]:


model2 = Wav2Vec2ForCTC.from_pretrained(output_models_dir).to("cuda")
processor2 = Wav2Vec2Processor.from_pretrained(output_models_dir)


# Now, we will just take the first example of the test set, run it through the model and take the `argmax(...)` of the logits to retrieve the predicted token ids.

# In[ ]:


input_dict = processor2(common_voice_test["input_values"][0], return_tensors="pt", padding=True)

logits = model2(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]


# We adapted `common_voice_test` quite a bit so that the dataset instance does not contain the original sentence label anymore. Thus, we re-use the original dataset to get the label of the first example.

# In[ ]:


common_voice_test_transcription = load_dataset("common_voice", language_code, data_dir=data_dir, split="test")


# Finally, we can decode the example.

# In[ ]:


print("Prediction:")
print(processor2.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription["sentence"][0].lower())


# Alright! The transcription can definitely be recognized from our prediction, but it is far from being perfect. Training the model a bit longer, spending more time on the data preprocessing, and especially using a language model for decoding would certainly improve the model's overall performance. 
# 
# For a demonstration model on a low-resource language, the results are acceptable, however 🤗.

# In[ ]:


import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
test_dataset = load_dataset("common_voice", "ga-IE", split="test")
wer = load_metric("wer")
def is_upper_vowel(letter):
    if letter in ['A', 'E', 'I', 'O', 'U', 'Á', 'É', 'Í', 'Ó', 'Ú']:
        return True
    else:
        return False
def irish_lower(word):
    if len(word) > 1 and word[0] in ['n', 't'] and is_upper_vowel(word[1]):
        return word[0] + '-' + word[1:].lower()
    else:
        return word.lower()
def irish_lower_sentence(sentence):
    return " ".join([irish_lower(w) for w in sentence.split(" ")])
chars_to_ignore_regex = '[,\?\.\!\;\:\"\“\%\‘\”\(\)\*]'
def remove_special_characters(sentence):
  tmp = re.sub('’ ', ' ', sentence)
  tmp = re.sub("’$", '', tmp)
  tmp = re.sub('’', '\'', tmp)
  tmp = re.sub(chars_to_ignore_regex, '', tmp)
  sentence = irish_lower_sentence(tmp) + ' '
  return sentence
resampler = torchaudio.transforms.Resample(48_000, 16_000)
# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = remove_special_characters(batch["sentence"])
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor2(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model2(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))

