from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from pathlib import Path
import pickle as pkl

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / 'model.p'

TRAIN_DIR = BASE_DIR / 'data' / 'train'

IMG_PATH = TRAIN_DIR / 'cloudy' /'cloudy1.jpg'


with open(MODEL_FILE.as_posix(), 'rb') as f:
    model = pkl.load(f)

img2vec = Img2Vec()

img = Image.open(IMG_PATH.as_posix())


features = img2vec.get_vec(img)

pred = model.predict([features])
print(pred)

