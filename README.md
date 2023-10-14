# General classificiton library

This is a setup for multi-class classification problems. I have used Pytorch Lightning to make the code more readable and easier to use.

## Basic CNN fine-tuning for classification

This is a basic setup for fine-tuning a CNN for classification. The code is written in Pytorch Lightning and is easy to use.

### Usage

```python
python train.py
```

The data should be symlinked into the `data` folder. The data should be organized as follows:

```
data/class_1/*.jpg
data/class_2/*.jpg
...
```

## CLIP zero-shot classification

This is a setup for zero-shot classification using CLIP. The code is written using the Huggingface Transformers library and is easy to use.

If you have Hungarian class names, you can use the `translate.py` script to translate it to English.

```python
python translate.py
```

### Usage

```python
python clip.py
```

This will output the ROC curves for the zero-shot classification in the `outputs` folder  as well as the confusion matrix. No training is required, only inference.

