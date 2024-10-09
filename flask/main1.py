from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>星期三,wayne您好!</h1>"

# 開發環境的指令 flask --app main1 run --debug
# run 正式產品的時候 gunicron main1:app