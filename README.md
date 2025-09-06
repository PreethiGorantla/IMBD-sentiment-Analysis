# IMBD-sentiment-Analysis
This project implements a sentiment analysis model on the IMBD movie reviews dataset using deep learning in Google colab.
This model classifies movie reviews as positive or negative using pre-trained GloVe embeddings and recurrent neural networks (LSTM & GRU).

🚀Features
Preprocessing of IMBD dataset (tokenization,padding,train-test split).
Deep Learning model built with Tensorflow/Keras (LSTM and GRU).
Word embeddings using **GloVe(Global Vectors for Word Representation)**
Sentiment prediction on unseen movie reviews.
Google colab notebook for easy reproducibility.

📊 Dataset
IMDB Movie Reviews Dataset
Contains 50,000 reviews labeled as positive or negative.
Available in tensorflow_datasets or keras.datasets.imdb.

🔤 Word Embeddings
Pre-trained GloVe embeddings (glove.6B.100d.txt) were used.
Each word is represented in a 100-dimensional vector space.
Helps capture semantic relationships (e.g., good ≈ great, bad ≈ terrible).

⚙️ Model Architectures
LSTM (Long Short-Term Memory):
Captures long-term dependencies in text sequences.
GRU (Gated Recurrent Unit):
A lighter, faster alternative to LSTM with comparable performance.
Both models were trained on GloVe-embedded sequences for binary classification (positive/negative).

⚙️ Requirements
numpy
pandas
matplotlib
scikit-learn
tensorflow

📈 Results
LSTM model: ~87% accuracy
GRU model: ~85% accuracy
Both models generalize well on unseen reviews.
Outperformed simple one-hot or randomly initialized embeddings.

📌 Future Improvements
🔍 Experiment with BERT/Transformers for higher accuracy.
🌐 Deploy the model as a web app (Flask/Streamlit).

 ✨ Author
 👩‍💻 Developed by Preethi Gorantla using Google Colab.
