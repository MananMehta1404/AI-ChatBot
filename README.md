# AI Chatbot with Flask and JavaScript


https://github.com/MananMehta1404/AI-ChatBot/assets/93565760/683e9508-1c47-4e0f-a43d-72ca5c7be3b8


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
