from collections import defaultdict
from functools import partial
from pathlib import Path
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.metrics import accuracy_score
import pickle as pkl
from libs.utils import get_files, get_dirs, move_percentage_to_test
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"

TEST_DIR = DATA_DIR / "test"

CLOUDY_DIR = TRAIN_DIR / "cloudy"

RAIN_DIR = TRAIN_DIR/ "rain"

SHINE_DIR = TRAIN_DIR/ "shine"

SUNRISE_DIR = TRAIN_DIR / "sunrise"

MODEL_FILE = BASE_DIR / "model.p"


img2Vec = Img2Vec()



data = defaultdict(partial(np.ndarray, 0))

for dir in get_dirs(DATA_DIR):
    features = []
    labels = []
    for category_dir in get_dirs(dir):
        for image_path in get_files(category_dir):
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_features = img2Vec.get_vec(img)
            features.append(img_features)
            labels.append(category_dir.stem.__str__())

    data[f'{dir.stem.__str__()}_data'] = np.asarray(features)
    data[f'{dir.stem.__str__()}_labels'] = np.asarray(labels)


model = RandomForestClassifier()



model.fit(data[f'{TRAIN_DIR.stem.__str__()}_data'], data[f'{TRAIN_DIR.stem.__str__()}_labels'])

with open(MODEL_FILE.as_posix(), 'wb') as f:
    pkl.dump(model, f)

y_pred = model.predict(data[f'{TEST_DIR.stem}_data'])

acc = accuracy_score(data[f'{TEST_DIR.stem}_labels'], y_pred)

print('Accuracy:', acc)





