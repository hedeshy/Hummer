# Setup
- Requires Python 3.8
- `pip3 install pipenv`
- Navigate to this directory
- `pipenv install`

# Execution
- `pipenv shell`
- Run corresponding scripts by calling `python "script".py`
- Optionally, run `pyinstaller --onefile --hidden-import='pkg_resources.py2_warn' recorder.py` to create an executable

# TODO
- make model possible from entire dataset
- k-cross-validation for dataset
- move model features etc. in own, reusable python module
- decide whether more than one humming mode label is required
- make WebSocket more efficient
- make `classifier.py` take more than one file