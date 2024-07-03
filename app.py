from flask import Flask, render_template, request, redirect, url_for
import os
from extract_features import predict_genre

app = Flask(__name__)

# Specify the template folder explicitly
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Function to ensure the 'uploads' folder exists
def ensure_upload_folder():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    ensure_upload_folder()  # Ensure the 'uploads' folder exists
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return redirect(url_for('index'))
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        predicted_genre = predict_genre(file_path)
        return render_template('index.html', predicted_genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
