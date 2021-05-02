from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import os

# TODO(developer)
# project_id = "your-project-id"
# subscription_id = "your-subscription-id"
# Number of seconds the subscriber should listen for messages
timeout = 5.0


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential_file.json'


subscriber = pubsub_v1.SubscriberClient()
# The `subscription_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path('genial-current-311020', 'sub_url') # pylint: disable=no-member

class subscribe_class:
    def get_msg_from_sub(self):

        msg = ''

        def callback(message):
            global msg
            msg += message['data']
            message.ack()
            # return msg

        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}..\n")
        

        # Wrap subscriber in a 'with' block to automatically call close() when done.
        with subscriber:
            try:
                # When `timeout` is not set, result() will block indefinitely,
                # unless an exception is encountered first.
                streaming_pull_future.result(timeout=timeout)
            except TimeoutError:
                streaming_pull_future.cancel()
        return msg
