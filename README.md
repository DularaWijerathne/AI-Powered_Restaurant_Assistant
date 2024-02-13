# AI-Powered Restaurant Assistant

This project implements a restaurant chatbot using Natural Language Processing (NLP) techniques in Python. The chatbot is designed to interact with users, answer questions, and provide assistance related to restaurant inquiries. The implementation involves four main components: text preprocessing, model training, chatbot creation, and intent management.

1. Text Preprocessing:
The `text_preprocessor.py` module handles the preprocessing of user input and training data. It utilizes the NLTK library for tasks such as tokenization and lemmatization. Tokenization breaks down sentences into individual words or units, while lemmatization reduces words to their base or root form for better analysis. The preprocessing steps prepare the text data for further processing and model training.

2. Model Training:
The `train.py` script trains a neural network model using TensorFlow and Keras. It constructs a neural network architecture suitable for classifying user intents based on input patterns. The training data, extracted from a JSON file containing intents and associated patterns, undergoes preprocessing and is used to train the model. The trained model learns to predict the intent of user queries during chatbot interactions.

3. Chatbot Creation:
The `chatbot.py` script creates the interactive chatbot interface. It loads the trained model and necessary data structures, such as token and tag dictionaries, for inference. The chatbot employs the trained model to predict the intent of user messages and generates appropriate responses based on predefined intents and associated responses stored in a JSON file. The chatbot engages in conversation with users, providing relevant information and assistance regarding restaurant-related inquiries.

4. Intent Management:
The `intents.json` file contains a collection of intents, each comprising a tag representing the intent category and associated patterns and responses. These intents serve as the basis for training the chatbot model and determining appropriate responses during interactions. The chatbot utilizes intent recognition to understand user queries and deliver contextually relevant responses.

Overall, this project demonstrates the application of NLP techniques and machine learning to develop a functional chatbot tailored for restaurant-related interactions. Through preprocessing, training, and intent management, the chatbot effectively interprets user messages and provides accurate responses, enhancing the user experience in restaurant inquiries and assistance.

