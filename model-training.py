# -*- coding: utf-8 -*-
"""Untitled43.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IOhaY8sczZ8snU0zsNm_xinhn08c1cGX
"""

!pip install -Uq fastai

from fastai.vision.all import *

!unzip drive/MyDrive/cycle.zip

!rm -r cycle/PP

path = "cycle"
dls = ImageDataLoaders.from_folder(path, valid_pct = 0.1, item_tfms=Resize(460),batch_tfms=aug_transforms(size=224))

dls.show_batch()

dls.show_batch()

learn = vision_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(15)

learn.export('model.pkl')

from google.colab import files
files.download('cycle/model.pkl')



