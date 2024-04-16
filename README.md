# AI Chatbot with Flask and JavaScript

## Initial Setup:

Clone repo and create a virtual environment
```
$ git clone https://github.com/MananMehta1404/AI-ChatBot
$ cd AI-ChatBot/flask_app
$ python3 -m venv venv
$ . venv/bin/activate
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Run
```
$ (venv) python app.py
```
Your flask app will be running on http://127.0.0.1:5000.
