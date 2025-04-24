from flask import Flask, render_template, redirect
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/start_recognition')
def start_recognition():
    script_path = os.path.join(os.path.dirname(__file__), "image_recognition.py")
    subprocess.run(["python", script_path])  # Adjust if using python3
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
