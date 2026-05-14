from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import sqlite3
import csv

app = Flask(__name__)
app.secret_key = "hackathon_secret"

model = load_model("model/waste_model.h5")

# ✅ Categories order fixed according to class_indices
categories = ["E-waste", "metal", "organic", "plastic"]

# ✅ Disposal suggestions updated to match categories
disposal_suggestions = {
    "E-waste": "Dispose at certified e-waste center ⚡",
    "metal": "Recycle or sell as scrap 🔧",
    "organic": "Compost or use as fertilizer 🌱",
    "plastic": "Send to recycling plant ♻️"
}

# ✅ Global list for recent predictions
recent_predictions = []

def get_db_connection():
    conn = sqlite3.connect("complaints.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("No file selected!", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    upload_folder = "backend/static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    # ✅ Save to recent predictions list
    recent_predictions.insert(0, {
        "image": file.filename,
        "prediction": predicted_class,
        "confidence": confidence
    })
    # Keep only last 5
    recent_predictions[:] = recent_predictions[:5]

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=confidence,
        image=file.filename,
        suggestion=disposal_suggestions[predicted_class],
        recent_predictions=recent_predictions
    )

# ✅ Complaint form route
@app.route("/complaint", methods=["GET", "POST"])
def complaint():
    if request.method == "POST":
        desc = request.form["description"]
        location = request.form["location"]
        conn = get_db_connection()
        conn.execute("INSERT INTO complaints (description, location, status) VALUES (?, ?, ?)",
                     (desc, location, "Pending"))
        conn.commit()
        conn.close()
        flash("Complaint submitted successfully!", "success")
        return redirect(url_for("dashboard"))
    return render_template("complaint.html")

# ✅ Dashboard route
@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    complaints = conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
    conn.close()
    total = len(complaints)
    resolved = sum(1 for c in complaints if c["status"] == "Resolved")
    pending = total - resolved

    bin_levels = {"E-waste": 20, "metal": 55, "organic": 70, "plastic": 40}

    return render_template("dashboard.html",
                           complaints=complaints,
                           total=total,
                           resolved=resolved,
                           pending=pending,
                           bin_levels=bin_levels)

# ✅ Resolve complaint route
@app.route("/resolve/<int:complaint_id>")
def resolve_complaint(complaint_id):
    conn = get_db_connection()
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", ("Resolved", complaint_id))
    conn.commit()
    conn.close()
    flash("Complaint marked as resolved!", "info")
    return redirect(url_for("dashboard"))

# ✅ Delete complaint route
@app.route("/delete/<int:complaint_id>")
def delete_complaint(complaint_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM complaints WHERE id = ?", (complaint_id,))
    conn.commit()
    conn.close()
    flash("Complaint deleted!", "danger")
    return redirect(url_for("dashboard"))

# ✅ Download report route
@app.route("/download_report")
def download_report():
    conn = get_db_connection()
    complaints = conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
    conn.close()

    filename = "complaints_report.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Description", "Location", "Status"])
        for c in complaints:
            writer.writerow([c["id"], c["description"], c["location"], c["status"]])

    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)