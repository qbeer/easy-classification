import torch
import open_clip
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from train import check_corrupted
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def single_image_run(image, model, preprocess, tokenizer, classes):
    image = preprocess(image).unsqueeze(0)
    text = tokenizer([description for description in classes])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return text_probs.numpy()[0]
    

def run(args):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    ds = ImageFolder('./data', transform=None, target_transform=None, is_valid_file=check_corrupted)
    classes = ds.classes
    
    import json
    with open('outputs/class2text.json', 'r') as f:
        class2text = json.load(f)
        
    english_classes = [class2text[class_name.lower().replace(' ', '').replace('_', ' ')] for class_name in classes]
    
    preds = []
    gts = []
    
    for image_dir in tqdm(os.listdir('./data')):
        for image in tqdm(os.listdir(os.path.join('./data', image_dir))):
            image_path = os.path.join('./data', image_dir, image)
            try:
                image = Image.open(image_path)
            except Exception as e:
                print('exception')
                print(e)
                continue
            y_hat = single_image_run(image, model, preprocess, tokenizer, english_classes)
            preds.append(y_hat)
            gts.append(ds.classes.index(image_dir))
            
    cm = confusion_matrix(gts, np.array(preds).argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=False)
    
    plt.savefig('outputs/valid_confusion_matrix.png', dpi=100)
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(20, 20))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(np.array(gts) == i, np.array(preds)[:, i])
        ax.plot(fpr, tpr, label=f'{classes[i]}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    
    plt.savefig('outputs/roc_curves.png', dpi=100)
            
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=5)
    
    args = parser.parse_args()
    exit(run(args))