import sys
import time
import hmac
import hashlib
import base64

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from HarlanBot_module import ChatBOT

if len(sys.argv) == 1:
    is_training = False
else:
    is_training = sys.argv[1] == '-train'

bot = ChatBOT(is_training,0)

app = Flask(__name__)


allowed_origins = [
    "https://harlansr.github.io",
    "http://127.0.0.1:5500", 
]
# CORS(app)
CORS(app, origins=allowed_origins)

@app.before_request
def middleware():
    if request.method == 'POST':
        x_signature = request.headers.get('X-SIGNATURE')
        if not x_signature:
            return jsonify({"status": "failed", "message": "Unauthorized"}), 401
        data_signature = x_signature.split('.')
        valid_1 = len(data_signature) != 3
        if valid_1:
            return jsonify({"status": "failed", "message": "Unauthorized"}), 401
        valid_2 = data_signature[0] != 'f496bf4066de4769a37c586eb61706b3'
        cur_time = int(time.time()) 
        valid_3 = not ((cur_time - int(data_signature[2]))<5 and cur_time >-1)
        
        message = str(data_signature[2]).encode('utf-8')  
        key = data_signature[0].encode('utf-8')  
        hmac_sha512 = hmac.new(key, message, hashlib.sha512)
        h_signature_s = base64.b64encode(hmac_sha512.digest()).decode('utf-8')

        valid_4 = data_signature[1] != h_signature_s
        
        if valid_2 or valid_3 or valid_4:
            return jsonify({"status": "failed", "message": "Unauthorized"}), 401

        


@app.route('/api/predict', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return jsonify({"message": "API is running"})
    else:
        try:
            question = request.get_json()
            
            if question["input"]:
                response = bot.ask(question["input"],need_accuracy=True)
                return jsonify({"status": "success", "answer": response[0], "accuracy": response[1]})
            
            else:
                return jsonify({"status": "failed", "answer": "Hello there! Do you have any questions about Harlan?"}),400
        except:
            return jsonify({"status": "failed", "message":"Something wrong"}),400
        
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=3091)