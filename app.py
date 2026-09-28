from flask import Flask, jsonify, render_template, request, send_from_directory, flash,redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import subprocess
import os
import sys
import shutil
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "uploads"
UNKNOWNN_FOLDER = "unknown_faces"
KNOWN_FOLDER = "face_database"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UNKNOWNN_FOLDER"] = UNKNOWNN_FOLDER
video_extension = {"mp4", "avi", "mov", "mkv", "webm", "wmv"}
image_extension = {"jpg", "jpeg", "img", "bmp", "svg", "img"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UNKNOWNN_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FOLDER, exist_ok=True)
os.makedirs("attendance", exist_ok=True)
os.makedirs("recordings", exist_ok=True)


def generate_thumbnail(video_path, thumbnail_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()  # grabs the first frame
    if success:
        cv2.imwrite(thumbnail_path, frame)
    cap.release()

def isVideo(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in video_extension

def isImage(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in image_extension

@app.route("/")
def home():
    return render_template("liveStream.html")

@app.route("/start_record", methods=["POST"])
def start_record():
    result = subprocess.run(["python", "face_recognition.py"], capture_output=True, text=True)
    return result.stdout or result.stderr

@app.route("/updateDatabase", methods=["POST"])
def updateDatabase():
    try:
        subprocess.run(["python", "build_database.py"])
        return "Database has been updated"
    except(RuntimeError):
        return "Database has failed to be updated"

@app.route("/uploadVidPage")
def uploadVidPage():
    video_list = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if isVideo(filename):
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            thumbnail_filename = os.path.splitext(filename)[0] + ".jpg"
            thumbnail_path = os.path.join(UPLOAD_FOLDER, thumbnail_filename)

            if not os.path.exists(thumbnail_path):
                generate_thumbnail(video_path, thumbnail_path)

            video_list.append({
                "title": os.path.splitext(filename)[0],  # filename without extension
                "video_file": filename,
                "thumbnail_file": thumbnail_filename
            })
    return render_template("uploadVid.html", videos=video_list)

@app.route("/liveStreamPage")
def liveStreamPage():
    return render_template("liveStream.html")

@app.route("/assignFacePage")
def assignFacePage():
    uFaceList = []
    knownList = []
    for face in os.listdir(UNKNOWNN_FOLDER):
        if isImage(face):
            uFacePath = os.path.join(UNKNOWNN_FOLDER, face)
            uFaceTitle = os.path.splitext(face)[0]

            uFaceList.append({
                "title" : uFaceTitle, #without .jpg and stuff
                "image" : face,       #with .jpg and stuff
                "path" : uFacePath
            })
    for known in os.listdir(KNOWN_FOLDER):
        knownList.append({
            "name" : known
        })
    return render_template("assignFace.html", images = uFaceList, knowns = knownList)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/unknown_faces/<path:filename>")
def detected_unknown(filename):
    return send_from_directory(app.config["UNKNOWNN_FOLDER"], filename)

@app.route("/uploadVid", methods=["POST"])
def uploadVid():
    file = request.files["vid"]
    if file.filename == "" or not isVideo(file.filename):
        return "Invalid file"
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    return f"File '{filename}' uploaded successfully!"

@app.route("/processVideo", methods=["POST"])
def processVideo():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")

    video_filename = next(
        (
            filename for filename in os.listdir(UPLOAD_FOLDER)
            if isVideo(filename) and os.path.splitext(filename)[0] == title
        ),
        None,
    )

    if video_filename is None:
        return jsonify({"error": "Video not found"}), 404

    video_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, video_filename))
    subprocess.Popen([sys.executable, "process_video.py", video_path])

    return jsonify({"message": f"Started processing '{title}'"})

@app.route("/AssignFace/<image>", methods=["POST"])
def assignFace(image):
    assignedPerson = request.form["person"]
    assignedPersonPath = os.path.join(KNOWN_FOLDER, assignedPerson)
    if os.listdir(assignedPersonPath) == []:
        imageName = assignedPerson + "0.jpg"
    else:
        latestPersonImage = sorted(os.listdir(assignedPersonPath), key=lambda p: int(re.search(r'\d+', p).group()))[-1]
        index = int(re.search(r'\d+', latestPersonImage).group())
        imageName = assignedPerson + str(index + 1) + "." + image.split(".")[1]
    assignedImagePath = os.path.join(KNOWN_FOLDER, assignedPerson, imageName)
    chosenImagePath = os.path.join(UNKNOWNN_FOLDER, image)
    shutil.move(chosenImagePath, assignedImagePath)
    flash("Face assigned successfully!")
    print("Uploaded:\nPerson:"+assignedPerson+"\nImage:"+image+"\nNew image name:"+imageName)
    return redirect(url_for("assignFacePage"))

@app.route("/AddPerson", methods=["POST"])
def AddPerson():
    name = request.form["name"]
    person_folder = os.path.join(KNOWN_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)
    return "New person created succesfully! Refresh the page to see it"

if __name__ == "__main__":
    app.run(debug=True)