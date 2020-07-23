from sklearn.ensemble import RandomForestClassifier
from joblib import load

clf = load('model.joblib')

print(clf)

# TODO
# Take microphone input
# Load model (done)
# Open WebSocket and send humming yes/no to Web
# Make minimal website that listens to the websocket
# https://websockets.readthedocs.io/en/stable/intro.html