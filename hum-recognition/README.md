# Setup
- Requires Python 3.8
- `pip3 install pipenv`
- Navigate to this directory
- `pipenv install`

# Execution
- `pipenv shell`
- To record the audio, run `python ./recorder.py`
- To compose the dataset, run `python ./dataset.py`. Considers recordings from `./data` folder, only. One must move recordings there.
- To train the model, run `python ./model.py`
- To use the model, run `python ./recognize.py`

# Make recorder standalone
Run `pyinstaller --onefile --hidden-import='pkg_resources.py2_warn' recorder.py` to create an executable

# TODO
- Optimize model parameters with grid search
- Refactor the code