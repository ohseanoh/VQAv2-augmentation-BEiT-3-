from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/seanoh/unilm/beit3/models/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/data/Shared_Data/VQAv2_aug",
    tokenizer=tokenizer,
    annotation_data_path="/data/Shared_Data/VQAv2_aug/vqa",
)
