from flask import Flask
from flask import render_template, request
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
PRE_TRAINED_MODEL_NAME = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 3)
model_path = r'E:\BanglaSent-SavedModels\saved-models\xlm_roberta_base_final.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model = model.to(device)
print('Model Loaded')

@app.route("/", methods=['Get'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    encoded_input = tokenizer(input_text, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    
    output = model(**encoded_input)
    _, pred = torch.max(output.logits, dim=1)
    pred = pred.detach().cpu().numpy()
    
    print(pred)
    if pred==0:
        return render_template('index.html', output='Neutral')
    elif pred==1:
        return render_template('index.html', output='Pro-Ukraine')
    elif pred==2:
        return render_template('index.html', output='Pro-Russia')
    else:
        return render_template('index.html', output='Unpredictable') 

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)