import numpy as np
import random
from tensorflow.keras.models import load_model


def test_model(model, test_number=100):
    for i in range(test_number):
        print(f"test number: {i}")
        input = np.array([[random.choice([True, False]) for _ in range(10)]])
        print(input)
        output = model.predict(input)
        print(output)
        print(f"score: {np.sum(output)}")


if __name__ == "__main__":
    # load model
    model = load_model('model.keras')
    # make test to show if it works
    test_model(model)
