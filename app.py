from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import warnings

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Decode base64 image from browser
        data = request.json['image']
        encoded = data.split(',')[1]
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        data_aux = []
        x_ = []
        y_ = []

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = float(max(prediction_proba[0]))
            predicted_character = labels_dict[int(prediction[0])]

            return jsonify({
                'prediction': predicted_character,
                'confidence': round(confidence * 100, 2)
            })

        return jsonify({'prediction': '', 'confidence': 0})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)