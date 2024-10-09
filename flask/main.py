# 開發環境的指令 flask --app main1 run --debug
# run 正式產品的時候 gunicron main1:app
# ctrl + c 是停止伺服器運作

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return '''<h1>這是python的project</h1>
            <p>這是在codespace環境開發的</p>
    '''