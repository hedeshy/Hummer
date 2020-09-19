# Setup
- Requires Python 3.8: <https://www.python.org/downloads/>
- `pip3 install pipenv`
- Navigate to this directory
- Run `pipenv install`

# Execution
- `pipenv shell`
- To record the audio for training the model, run `python ./recorder.py`. Files are stored in this directory.
- To compose the dataset for training the model, run `python ./dataset.py`. Considers recordings from `./data` directory, only. One must move recordings there.
- To train the model, run `python ./model.py`
- To use the model for real-time recognition of humming, run `python ./recognize.py`

# Make recorder standalone
Run `pyinstaller --onefile --hidden-import='pkg_resources.py2_warn' recorder.py` to create an executable

# TODO
- Optimize model hyper parameters with grid search
- Make recorder interface with a Web page and connect via WebSocket