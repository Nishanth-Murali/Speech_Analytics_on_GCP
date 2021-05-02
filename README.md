# Speech_Analytics_on_GCP

In this project, we aim to build a cloud-application that analyses the speeches, which can be an audio file or a transcript, given as input by the users. The analyses include: 
1. Topic modelling: This tells us about the topics the speaker has talked about.
2. Sentiment analysis: We can know how positive/negative the sentiments of the speaker are.

This application can be used in/for;
1. Recommendation Engines: The NLP model can be used to recommend similar speakers to the users. 
2. Political speech analysis: The application can be used by media houses to analyze the speeches given by politicians.

SYSTEM ARCHITECTURE:

![alt text](https://github.com/Nishanth-Murali/Speech_Analytics_on_GCP/blob/main/System_design_CC.jpeg)

Following are the components of Google Cloud, used in our project:
Google Load Balancer: This component works as a traffic filter which will be the only component visible to the public, for sending HTTP requests.
Google App Engine (GAE):
1.	Front-end: A Flask application is run on the GAE, which together forms a Virtual Machine, which will be the UI to take in inputs from the user and give the outputs back accordingly.
2.	Back-end: Again, a Flask server that runs as a listener on another GAE, which will lift all the workload of processing the inputs using the Natural Language Processing (NLP) model. These instances will be auto-scaled, according to the number of inputs given by the user.
Google Pub/Sub: 
The front-end server sends the URLs of the transcripts, that are stored in the google cloud storage, to the Publisher. These URLs are subscribed by the Subscriber at the back-end GAE that would start running the model for the given transcript (URL). This would also help in decoupling the front-end from the back-end, so that asynchrony is maintained.


Google Cloud storage:
Input storage:
The inputs from the user will be uploaded to a google cloud storage’s bucket and the blob’s URL will be published to the publisher. 
Output storage:
The images obtained by running the NLP model on the GAE will be stored in another bucket, in order to achieve persistent storage. These images will be picked up by the front-end GAE, which will render these images on the UI, to be viewed by the user.
VPC:
It acts like a network confinement for the application, so that only the authorized users can access the application and thus making it secure.

CLOUD SERVICES (APIs) USED:
Cloud storage:
This API will be used to create, access and delete user inputs and the outputs that are to be sent back to the UI. The ‘gsutil’ URLs of these objects will be used to engage the Pub/Sub API, so that the front-end and the back-end GAEs are decoupled.
Google Pub/Sub:
The Pub/Sub API is used as the bridge between the app-server and the web-server. The input images’ ‘gsutil’ URLs will be used to publish to the Publisher and they will be subscribed by the Subscriber. This is done to ensure that both the servers (GAEs) used in the application work in an asynchronous fashion and are also decouple to enhance the performance of the system as a whole.
Google Cloud Speech:
This handy API is used to recognize the speech that will be recorded live by speaker/user. The output obtained from the API will be a transcript text, which will be analyzed by the NLP model in the back-end.
Google Cloud Translate:
This is used to translate the transcripts in different languages, given by the user, to English. These will also be fed to the NLP model in order to be processed and analyzed.

How to install and run the application on Google Cloud Platform?

We have used Python 3.9 as the runtime for the application.
For the front-end Flask app to run and deploy on GAE;
1.	Open the ‘FlaskProject’ folder in an IDE and type the following command:
      gcloud app deploy

2.	To open the app in a web browser, type;
      gcloud app browse 
and click the URL that shows up.

For the back-end Flask app to run and deploy on GAE;
3.	Open the ‘NLP_model’ folder in an IDE and type the following command:
      gcloud app deploy

4.	To open the app in a web browser, type;
      gcloud app browse 
and click the URL that shows up.

Thus, we can have both, the app-server and web-server instances up and running for the user to give the inputs through the UI and get back the appropriate outputs.

References:

[1] Programming Google App Engine with Python [Book] by O’Reilly.

[2] Learning document embeddings along with their uncertainties Edit social preview; Santosh        Kesiraju, Oldřich Plchot, Lukáš Burget, Suryakanth V. Gangashetty.

[3] Neural Variational Inference for Text Processing Edit social preview; Yishu Miao, Lei Yu, Phil Blunsom.
