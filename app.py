from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
import base64
from dotenv import load_dotenv
import json

load_dotenv()

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/faceLogin':
            print("here")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            nparr = np.fromstring(base64.b64decode(post_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            userId = 1
            data = {'userId': userId}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))


host = '192.168.0.107'
port = 3002
httpd = HTTPServer((host, port), MyRequestHandler)
print(f"Сервер запущен на http://{host}:{port}")
httpd.serve_forever()