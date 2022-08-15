from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories = ('HDPE', 'PET')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label()
examples = ['ex1.jpg', 'ex2.jpg', 'ex3.jpg', 'ex4.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)