from flask import Flask
import pre_training
from os.path import abspath, dirname
from pathlib import Path

app = Flask(__name__)
app.register_blueprint(pre_training.bp)
#
# @app.route("/")
# def hello():
#     return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)