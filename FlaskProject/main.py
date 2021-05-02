import json
import os
from datetime import time, datetime, timezone
from random import random
from google.cloud import storage
from speech_to_text import speech_test
from Publish import publish_class


from flask import Flask, render_template, request, Response, jsonify

from multiprocessing import Process



app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential_file.json'



st = speech_test()
pc = publish_class()


@app.route('/')
def hello_world():
    return \
        render_template("HomePage.html", title="SpeechMiner")


@app.route('/upload', methods=['POST'])
def upload_file():
    print('here')
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        for file in files:
            filename = str(random()) + '.wav'
            with open(filename, 'wb') as audio:
                file.save(audio)
            upload_blob('persistent-input-store', filename, filename)
            gcs_uri = "gs://persistent-input-store/" + filename
            # pc.pub_message(gcs_uri)
            st.speech_to_txt(gcs_uri)

            # upload_blob('persistent-input-store', filename, filename)
    return render_template("HomePage.html", title="SpeechMiner")


@app.route('/transcript', methods=['POST'])
def transcript():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        for file in files:
            print("here")
            filename = str(random()) + '.txt'
            file.save(filename)
            upload_blob('persistent-input-store', filename, filename)
            # gcs_uri = "gs://persistent-input-store/" + filename
            # pc.pub_message(gcs_uri)
            upload_blob('transcript-input-bucket-6344', filename, filename)
            text_file_name =  filename
            pc.pub_message(text_file_name)
            i = 1000000000
            while(i > 0):
                i -= 1
    return render_template("HomePage.html", title="SpeechMiner")



@app.route("/rec", methods=['POST', 'GET'])
def record():
    if request.method == "POST":
        filename = str(random()) + '.wav'
        f = request.files['audio_data']
        with open(filename, 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        upload_blob('persistent-input-store', filename, filename)
        gcs_uri = "gs://persistent-input-store/" + filename
        # pc.pub_message(gcs_uri)
        st.speech_to_txt(gcs_uri)
        return render_template('HomePage.html', request="POST")
    else:
        return render_template("HomePage.html", title="SpeechMiner")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


@app.route("/output", methods=['POST', 'GET'])
def output():
    storage_client = storage.Client()
    bucket = storage_client.bucket("image-output-bucket-6344")
    responseJson = {}
    count = 0
    for file in bucket.list_blobs():
        count = count + 1
        name = file.name
        blob = bucket.blob(file.name)
        url_lifetime = int(datetime.now(tz=timezone.utc).timestamp()) + 36000  # Seconds in an hour
        serving_url = blob.generate_signed_url(url_lifetime)
        responseJson[name] = serving_url
    
    if count == 0:
        return jsonify({"none":"none"})
    return jsonify(responseJson)


if __name__ == '__main__':
    app.run(debug=True)
    
