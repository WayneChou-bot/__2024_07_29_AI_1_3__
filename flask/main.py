from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"
    
# 開發環境的指令 flask --app main1 run --debug
# run 正式產品的時候 gunicron main1:app
# ctrl + c 是停止伺服器運作