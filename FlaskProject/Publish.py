"""Publishes multiple messages to a Pub/Sub topic with an error handler."""
import time

from google.cloud import pubsub_v1
import os

# TODO(developer)
# project_id = "your-project-id"
# topic_id = "your-topic-id"


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './credential_file.json'

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('genial-current-311020', 'pub_url') # pylint: disable=no-member

futures = dict()

class publish_class:
    def get_callback(self, f, data):
        def callback(f):
            try:
                print(f.result())
                futures.pop(data)
            except:  # noqa
                print("Please handle {} for {}.".format(f.exception(), data))

        return callback

    def pub_message(self, message):
        data = message
        futures.update({data: None})
        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data.encode("utf-8"))
        futures[data] = future
        # Publish failures shall be handled in the callback function.
        future.add_done_callback(self.get_callback(future, data))

    # Wait for all the publish futures to resolve before exiting.
        while futures:
            time.sleep(5)

        print(f"Published messages with error handler to {topic_path}.")
