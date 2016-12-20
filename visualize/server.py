from __future__ import print_function
import os
import json
import random

def read_data(data_path):

    datasets = dict()
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        with open(path, "r") as f:
            data = json.load(f)
            print(data["model"])
        datasets[filename] = data
    return datasets

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Visualize neural networks.')
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    datasets = read_data(args.data) 

from flask import Flask
app = Flask(__name__)

from flask import render_template

model_template = "type: {type}, dims: {dimSize}, layers: {layerSize}, attention: {attention}, datafile: {path}"

@app.route('/')
def model_picker():
    models = list()
    for model, data in datasets.items():
        info = model_template.format(path=data["path"],
                **data["model"])
        models.append({"name": model, "info": info})
    return render_template("model-picker.html", models=models)

@app.route('/model/<model>/example/<example>/epoch/<epoch>')
def display_example(model, example, epoch):
    print("displaying: ", model, "example=", example, "epoch=", epoch)
    example = int(example)
    epoch = int(epoch)
    data = datasets[model]

    input = data["examples"][example]["encoderInput"]
    gold_output = data["examples"][example]["goldDecoderOutput"]
    gold_output_json = json.dumps(gold_output)

    greedy_decoder_output = json.dumps(
        data["examples"][example]["models"][epoch]["greedyDecoderOutput"]) 
    beam_decoder_output = json.dumps(
        data["examples"][example]["models"][epoch]["beamDecoderOutput"]) 

    all_epochs = range(len(data["examples"][example]["models"]))
    all_examples = range(len(data["examples"]))

    if data["model"]["attention"]:
        attention = json.dumps(
            data["examples"][example]["models"][epoch]["attention"]) 
        template = "default.html"
    else:
        attention = None
        template = "no-attention.html"
    
    return render_template(template, 
            input=input, gold_output=gold_output,
            greedy_decoder_output=greedy_decoder_output,
            gold_output_json=gold_output_json,
            beam_decoder_output=beam_decoder_output, 
            current_model=model,
            current_example=example,
            all_examples=all_examples, 
            current_epoch=epoch,
            all_epochs=all_epochs, 
            attention=attention)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
