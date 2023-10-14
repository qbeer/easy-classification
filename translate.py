from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from torchvision.datasets import ImageFolder
from train import check_corrupted

def translate(text, model, tokenizer):
    # translate Hungarian to French
    tokenizer.src_lang = "hu"
    encoded_hu = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hu, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Hungarian: {text:50s} ----> English: {translation[0]}")
    
    return translation[0]

def run():
    ds = ImageFolder('./data', transform=None, target_transform=None, is_valid_file=check_corrupted)
    classes = ds.classes
    classes = [class_name.lower().replace(' ', '').replace('_', ' ') for class_name in classes]
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-hu-en")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-hu-en")
    
    class2text = {}
    
    for class_name in classes:
        translation = translate(f" Ez egy vegyészetben használt {class_name}. ", model, tokenizer)
        class2text[class_name] = translation
    
    import json
    with open('outputs/class2text.json', 'w') as f:
        json.dump(class2text, f, indent=4)
        
        
if __name__ == "__main__":
    exit(run())