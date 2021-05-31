from flask import Flask, request, jsonify
import pickle
import flask
import pandas as pd

app = Flask(__name__, template_folder='templates')


from Preprocess import make_pred
@app.route('/', methods = ['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))


    if flask.request.method == 'POST':
        Input = flask.request.form['Question']
        prediction = make_pred(Input)

    return flask.render_template('index.html', original_input= {'Question':Input}, result = prediction)





if __name__ == '__main__':
    app.run()
