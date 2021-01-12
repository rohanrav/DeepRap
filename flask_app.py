from flask import Flask, render_template, jsonify, request, flash
from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import re
import numpy as np
import os

maxlen = 100

def prepModel():
	text = dataPrep()
	(unique_chars, char_to_index_mapping, index_to_character_mapping) = createMappingsAndAlphabet(text)
	model = loadModel()

	return (model, char_to_index_mapping, index_to_character_mapping, unique_chars)

def dataPrep():
	os.environ['KMP_DUPLICATE_LIB_OK']='True'
	data = pd.read_csv('data/eminem_data.csv', encoding = "ISO-8859-1")

	lyrics = data['text']

	text = ""

	for i in range(0, len(lyrics)):
	    text = text + str(lyrics.iloc[i])

	text = re.sub('[#$():;*+-/]', '', text)

	return text

def createMappingsAndAlphabet(text):
	unique_chars = sorted(list(set(str(text))))
	print("total chars: ", len(unique_chars))

	char_to_index_mapping = dict((c, i) for i, c in enumerate(unique_chars))
	index_to_character_mapping = dict((i, c) for i, c in enumerate(unique_chars))

	return unique_chars, char_to_index_mapping, index_to_character_mapping

def loadModel():
	filepath = 'weights-eminem/weights-100-New.h5'
	model = load_model(filepath)

	return model

(model, char_to_index_mapping, index_to_character_mapping, unique_chars) = prepModel()

def predictText(seed):
	print('making predictions')

	generated = seed
	original_seed = seed

	for i in range(400):
		x_pred = np.zeros((1, maxlen, len(unique_chars)), dtype=np.bool)
		for t, char in enumerate(seed):
			x_pred[0, t, char_to_index_mapping[char]] = 1
    
		predicted = model.predict(x_pred, verbose=1)[0]
		next_char = np.argmax(predicted)
		char_pred = index_to_character_mapping[next_char]
		    
		generated += char_pred
		seed  = seed[1:]
		seed += char_pred

	print("\n------------------------ Generated Eminem Lyrics: -------------------------")
	print(generated)
	return generated


app = Flask(__name__)
app.config['SECRET_KEY'] = '063b30152ab0fe3bce2c3b9e39559b98'


@app.route('/', methods=['GET', 'POST'])
def hello():
	if request.method == 'POST':
		result = request.form["Name"]
		result = re.sub('[-,\n\r]', ' ', result)[0:100]
        
		if (result == "" or len(result) < 100):
			print("empty result")
			response = {
				'Name' : "Please enter a seed of 100 characters in length."
			}
		else:
			pred = predictText(str(result))
			response = {
				'Name' : ("Generated lyrics: " + pred)
			}

		return render_template("home.html", result = response)
	else:
		result = {
			'Name' : ""
		}
		return render_template("home.html", result = result)


if __name__ == '__main__':
	app.run(debug=True)
