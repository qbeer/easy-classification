from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from torchvision.datasets import ImageFolder
from train import check_corrupted

def translate(text, model, tokenizer):
    # translate Hungarian to French
    tokenizer.src_lang = "hu"
    encoded_hu = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hu, forced_bos_token_id=tokenizer.get_lang_id("en"))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    
    print(f"Hungarian: {text:25s} ----> English: {translation}")
    
    return translation

def run():
    ds = ImageFolder('./data', transform=None, target_transform=None, is_valid_file=check_corrupted)
    classes = ds.classes
    classes = [class_name.lower().replace(' ', '').replace('_', ' ') for class_name in classes]
    
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    
    for class_name in classes:
        translate(f"{class_name}", model, tokenizer)
        
if __name__ == "__main__":
    exit(run())