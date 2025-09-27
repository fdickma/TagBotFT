#!/bin/sh
python -m venv env
env/bin/pip install pandas sqlalchemy openpyxl
env/bin/pip freeze > requirements.txt
env/bin/pip install --upgrade -r requirements.txt
