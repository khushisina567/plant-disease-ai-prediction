# AI & ML Based Plant Disease Prediction and Cure

A Flask + TensorFlow project that accepts a plant leaf image, predicts a disease class, stores the prediction history, and displays symptoms, cause, treatment/care and prevention guidance.

## Run the demo
1. Install Python 3.10/3.11.
2. Create a virtual environment:
   python -m venv venv
3. Activate it:
   Windows: venv\Scripts\activate
4. Install packages:
   pip install -r requirements.txt
5. Start:
   python app.py
6. Open http://127.0.0.1:5000

## Train a real model
Download a suitable plant-leaf image dataset such as PlantVillage, arrange it as:
dataset/Class_Name/image.jpg
Then run:
python train_model.py

The trained model and class list are saved under model/. Restart Flask and predictions will use the trained model.

## Important
The treatment text is educational guidance, not a pesticide prescription. Follow local agricultural-extension advice and product labels before applying any treatment.
