from flask import Flask, render_template, Response, request, jsonify
import os
from detection import generate_frames, load_database_embeddings

app = Flask(__name__)
DATABASE_PATH = "my_database"

SAVE_DIR = "my_database"

@app.route('/') #Root 
def homepage():
    return render_template('homepage.html')

@app.route('/capturing')
def capture():
    return render_template('capturing.html')

@app.route("/detection")
def realtime():
    global embeddings, names  
    embeddings, names = load_database_embeddings(DATABASE_PATH)  # Reload embeddings
    return render_template("detection.html")

@app.route('/save_image', methods=['POST'])
def save_image():
    name = request.form['name'] 
    image_data = request.files['image'] 

    person_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    image_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.jpg") 
    image_data.save(image_path)

    return jsonify({"message": "Image saved successfully", "path": image_path}) 

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')  # New route for the About page
def about():
    return render_template('about.html') 

if __name__ == '__main__':
    app.run(debug=True)
