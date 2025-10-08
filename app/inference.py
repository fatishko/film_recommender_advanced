import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_input(user_prompt):
    pred = model.predict(user_prompt)
    return pred
