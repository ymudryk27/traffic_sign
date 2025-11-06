from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "secret"

@app.get("/")
def index():
    return render_template("index.html", ready=False)

@app.post("/predict")
def predict():
    
    flash("The model is not connected yet. This is just a test message.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)