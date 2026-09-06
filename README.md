# Smart Waste Management System

An AI-powered waste classification system that identifies waste as **Organic, Plastic, Metal, or E-waste** from an image and suggests the correct disposal method. Built end-to-end — from data preparation and model training to a Flask backend serving predictions and a dashboard for tracking classification history.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-orange)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)

<!-- Add a screenshot or GIF of the dashboard/prediction here — this matters a lot, drop it in before linking this repo anywhere -->
<!-- ![Demo Screenshot](assets/demo.png) -->

## Overview

Improper waste segregation is a major bottleneck in effective recycling and disposal. This project uses a fine-tuned **MobileNetV2** image classification model to automatically sort waste into four categories and recommend how each type should be disposed of, reducing manual sorting effort and human error.

## Features

- 🖼️ **Image-based waste classification** into Organic, Plastic, Metal, and E-waste
- ♻️ **Disposal suggestions** tailored to the predicted category
- 📊 **Dashboard** to view classification history
- 🗄️ **SQLite database** for persisting predictions and records
- 🧠 **Transfer learning** on MobileNetV2 for efficient, lightweight inference

## Tech Stack

| Layer         | Technology                                           |
| Backend       | Flask (Python)                                       |
| ML Model      | TensorFlow / Keras — MobileNetV2 (transfer learning) |
| Database      | SQLite                                               |
| Data Handling | scikit-learn (train/val/test split)                  |

## Model Performance

The model achieves **~92% accuracy** overall on the test set, with strong performance on **Plastic** and **Organic** waste. Accuracy is comparatively lower on **E-waste** and **Metal**, likely due to smaller/less varied training samples for those classes — a known limitation and a clear next step for improvement (see below).

## Project Structure

```
smart-waste-management/
├── backend/            # Flask app — routes, dashboard, prediction serving
├── model/              # Trained model artifacts
├── split_datset.py     # Splits raw dataset into train/val/test
├── train_model.py      # Trains the MobileNetV2 classifier
├── test_model.py       # Evaluates model performance on test data
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/monikagurbani444-afk/smart-waste-management.git
cd smart-waste-management
pip install -r requirements.txt
```

### Training the Model (optional — pretrained model included)

```bash
python split_datset.py
python train_model.py
python test_model.py
```

### Running the App

```bash
cd backend
python app.py
```

Then open `http://localhost:5000` in your browser.

## How It Works

1. User uploads an image of a waste item through the web interface.
2. The MobileNetV2 model classifies it into one of four categories.
3. The backend returns the predicted category along with a disposal suggestion.
4. The prediction is logged to SQLite and viewable in the dashboard's classification history.

## Limitations & Future Work

- Accuracy on E-waste and Metal classes trails Plastic and Organic — expanding and diversifying training data for these categories is the top priority.
- No live deployment yet — deploying on Render/Railway with a public demo link is a planned next step.
- Currently supports single-image classification; batch upload and real-time camera input are potential extensions.

## Author

**Monika Gurbani**
[GitHub](https://github.com/monikagurbani444-afk)
