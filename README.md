
# 🌦️ Weather Classification

This project performs image classification on weather categories (e.g., cloudy, rain, shine, sunrise) using feature extraction and a Random Forest classifier.

## 🧠 Overview

The pipeline:
1. Organizes raw weather images into category folders based on filename keys.
2. Splits training and test data by a defined percentage.
3. Extracts image features using `img2vec-pytorch`.
4. Trains a `RandomForestClassifier`.
5. Evaluates accuracy on the test set.

## 📁 Project Structure

```

weather\_classification/
│
├── data/
│   ├── train/              # Training images by category
│   └── test/               # Testing images by category
│
├── libs/
│   └── utils.py            # Helpers to load files and directories
│
├── main.py                 # Main training & evaluation script
└── README.md               # You are here!

````

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

You’ll need:

* `numpy`
* `Pillow`
* `scikit-learn`
* `img2vec-pytorch`

> ⚠️ Note: `img2vec-pytorch` requires PyTorch and a compatible torchvision model.

# You can train a new dataset or use the model

## Train a new dataset

### 1. Organize Data

If your dataset is in a single folder (e.g., `data/raw`), you can sort them by filename keys:

```python
from your_script import separate_images

separate_images(data_dir=Path('./data/raw'), keys=['cloudy', 'rain', 'shine', 'sunrise'])
```

This will move images to `data/train/<category>/`.

### 2. Split Data into Train/Test

Move a percentage of images to the test set:

```python
from your_script import move_percentage_to_test

move_percentage_to_test(data_dir=Path('./data'), keys=['cloudy', 'rain', 'shine', 'sunrise'], percentage=0.2)
```

### 3. Train and Evaluate

Run the main script:

```bash
python main.py
```

You will see output like:

```
Moved image1.jpg -> data/test/cloudy
...
Accuracy: 0.965
```
## 🧪 Use the Model

Once the model is trained, you can use it to make predictions on new images.

An example usage is provided in [`infer.py`](./infer.py).

### 🔍 Example: Making a Prediction

```python
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from pathlib import Path
import pickle as pkl

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / 'model.p'
TRAIN_DIR = BASE_DIR / 'data' / 'train'

# Sample image path
IMG_PATH = TRAIN_DIR / 'cloudy' / 'cloudy1.jpg'

# Load trained model
with open(MODEL_FILE.as_posix(), 'rb') as f:
    model = pkl.load(f)

# Extract image features
img2vec = Img2Vec()
img = Image.open(IMG_PATH.as_posix())
features = img2vec.get_vec(img)

# Make prediction
pred = model.predict([features])
print(pred)
```


## 📊 Result

The final classification accuracy is printed after evaluation. Accuracy may vary depending on image quality and balance.

## 👤 Author

**Iaggo Capitanio**
