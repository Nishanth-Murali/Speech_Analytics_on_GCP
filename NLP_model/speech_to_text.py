from google.cloud import speech
from google.cloud import translate
from random import random
from google.cloud import storage
import six
from Publish import publish_class

import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential_file.json'
# Instantiates a client
client = speech.SpeechClient()
pc = publish_class()

class speech_test:

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )



    # The name of the audio file to transcribe
    # gcs_uri = "gs://transcript-input-bucket-6344/0.04829222558524049.wav"

    def speech_to_txt(self, gcs_uri):
        audio = speech.RecognitionAudio(uri=gcs_uri)

        print(audio)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-US",
            audio_channel_count=1
        )

    # Detects speech in the audio file
        response = client.recognize(config=config, audio=audio)
        transcript_str = ""
        for result in response.results:
            transcript_str += result.alternatives[0].transcript

        tf_name = str(random()) + '.txt'
        with open(tf_name, 'wt') as text_file:
            text_file.write(transcript_str)
        #text_file.save(text_file)
        self.upload_blob('transcript-input-bucket-6344', tf_name, tf_name)
        text_file_name =  tf_name
        pc.pub_message(text_file_name)
