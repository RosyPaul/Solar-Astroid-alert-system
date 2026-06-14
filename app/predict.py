import joblib
import json
import numpy as np
import os


# Get project root (one level up from app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model, encoder and features
model = joblib.load(os.path.join(BASE_DIR, 'model', 'flare_model.joblib'))
le = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.joblib'))

with open(os.path.join(BASE_DIR, 'model', 'features.json'), 'r') as f:
    features = json.load(f)

ALERT_THRESHOLD = 0.4  # lower threshold for safety

def predict_flare(input_data: dict):
    input_data['flare_class_encoded'] = int(le.transform([input_data['flare_class']])[0])
    feature_values = [input_data[f] for f in features]
    X = np.array(feature_values).reshape(1, -1)
    probability = float(model.predict_proba(X)[0][1])
    prediction = 1 if probability >= ALERT_THRESHOLD else 0
    return {
        "prediction": prediction,
        "label": "HAZARDOUS ⚠️" if prediction == 1 else "SAFE ✅",
        "probability": round(probability, 4)
    }

if __name__ == "__main__":
    test_extreme = {
        'flare_class': 'X',
        'flare_intensity': 8.0,
        'duration_minutes': 120.0,
        'activeRegionNum': 12983.0,
        'has_linked_events': 1,
        'hour': 14,
        'month': 6,
        'dayofweek': 2
    }
    print("Test - Extreme flare:")
    print(predict_flare(test_extreme))

    # Test 2 - Weak A class flare
    test_weak = {
        'flare_class': 'A',
        'flare_intensity': 1.0,
        'duration_minutes': 10.0,
        'activeRegionNum': 0,
        'has_linked_events': 0,
        'hour': 6,
        'month': 1,
        'dayofweek': 0
    }
    print("\nTest 2 - Weak A class flare:")
    print(predict_flare(test_weak))

    # Test 3 - Moderate C class flare
    test_moderate = {
        'flare_class': 'C',
        'flare_intensity': 5.0,
        'duration_minutes': 30.0,
        'activeRegionNum': 12983.0,
        'has_linked_events': 0,
        'hour': 10,
        'month': 3,
        'dayofweek': 1
    }
    print("\nTest 3 - Moderate C class flare:")
    print(predict_flare(test_moderate))

    # Test 4 - Strong M class with linked events
    test_strong = {
        'flare_class': 'M',
        'flare_intensity': 7.0,
        'duration_minutes': 90.0,
        'activeRegionNum': 13200.0,
        'has_linked_events': 1,
        'hour': 18,
        'month': 9,
        'dayofweek': 4
    }
    print("\nTest 4 - Strong M class with linked events:")
    print(predict_flare(test_strong))
    # Test 5 - B class with linked events
    test_b = {
        'flare_class': 'B',
        'flare_intensity': 9.0,
        'duration_minutes': 60.0,
        'activeRegionNum': 13000.0,
        'has_linked_events': 1,
        'hour': 12,
        'month': 7,
        'dayofweek': 3
    }
    print("\nTest 5 - B class with linked events:")
    print(predict_flare(test_b))

    # Test 6 - X class but no linked events, short duration
    test_x_weak = {
        'flare_class': 'X',
        'flare_intensity': 1.1,
        'duration_minutes': 10.0,
        'activeRegionNum': 0,
        'has_linked_events': 0,
        'hour': 3,
        'month': 2,
        'dayofweek': 0
    }
    print("\nTest 6 - X class but weak signature:")
    print(predict_flare(test_x_weak))

    # Test 7 - M class no linked events
    test_m_alone = {
        'flare_class': 'M',
        'flare_intensity': 2.0,
        'duration_minutes': 25.0,
        'activeRegionNum': 12500.0,
        'has_linked_events': 0,
        'hour': 8,
        'month': 11,
        'dayofweek': 2
    }
    print("\nTest 7 - M class no linked events:")
    print(predict_flare(test_m_alone))