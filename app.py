from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16 MB

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the image-to-text pipeline
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ''
    image_url = ''
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')

        # Generate caption
        captions = image_to_text(image)
        caption = captions[0]['generated_text'] 
        image_url = f"/{app.config['UPLOAD_FOLDER']}/{filename}"  # Prepare the URL for the image

    return render_template('index.html', caption=caption, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
