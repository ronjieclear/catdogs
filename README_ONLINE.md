# Dog and Cat CNN Web App

Upload the contents of this `online` folder to your Python web server.

## Main Files

- `app.py` - Flask web application.
- `web_app.py` - Same Flask app, kept with the original project name.
- `dog_cat_tensorflow_model.weights.h5` - trained CNN weights.
- `templates/index.html` - web page.
- `static/styles.css` - page design.
- `static/uploads/` - uploaded images are saved here.
- `requirements.txt` - Python packages to install on the server.
- `Procfile` - tells web app hosts to run `gunicorn app:app`.
- `passenger_wsgi.py` - entry file for cPanel/Passenger Python apps.

## Install

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

## Run With Gunicorn

```bash
gunicorn app:app
```

## Important

The app uses TensorFlow, so the online server must support TensorFlow. Some cheap shared hosting plans do not allow TensorFlow because it is large.

The training and test images are not included in this online folder. The page shows the saved metrics from the latest local training run.
