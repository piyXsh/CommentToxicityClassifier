import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization


# Function to load Random Forest model and vectorizer for each label
def load_rf_model(label):
    with open(f'{label}_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(f'{label}_vect.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


# Function to load Bidirectional LSTM model and tokenizer
def load_bilstm_model():
    
    # Load the saved model
    model = load_model('bilstm.h5')
    
    # Load the saved vocabulary
    with open("tv_vocab.pkl", "rb") as f:
        loaded_vocab = pickle.load(f)

    # Recreate the TextVectorization layer
    vec = TextVectorization(max_tokens=len(loaded_vocab),
                                   output_mode='int',
                                   output_sequence_length=1800)

    # Set the vocabulary to the new vectorizer
    vec.set_vocabulary(loaded_vocab)
    

    return model, vec

# Streamlit App
st.title("Comment Toxicity Classifier")

# Dropdown to select model
model_choice = st.selectbox('Choose Model', ('Random Forest', 'Bidirectional LSTM'))

# Get user input for the comment
user_input = st.text_area("Enter a comment:")

# Define labels (Replace with your actual labels)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

if model_choice == 'Random Forest':
    st.subheader("Random Forest Model Selected")

    # Load all Random Forest models and vectorizers for each label
    models_vectorizers = {label: load_rf_model(label) for label in labels}
    
    if st.button('Classify with Random Forest'):
        if user_input:
            predictions = {}
            for label, (model, vectorizer) in models_vectorizers.items():
                vectorized_input = vectorizer.transform([user_input])
                predictions[label] = model.predict(vectorized_input)[0]
            st.write('Random Forest Predictions:')
            st.json(predictions)
        else:
            st.write("Please enter a comment to classify.")

elif model_choice == 'Bidirectional LSTM':
    st.subheader("Bidirectional LSTM Model Selected")

    # Load LSTM model and tokenizer
    lstm_model, token = load_bilstm_model()

    if st.button('Classify with Bidirectional LSTM'):
        if user_input:
            # Tokenize and pad the input
            vectorized_text = token([user_input])  # Wrap user_input in a list
            
            # LSTM requires a 3D input, so we need to reshape
            #vectorized_text = tf.expand_dims(vectorized_text, axis=0) 
            prediction = lstm_model.predict(vectorized_text)

            # Converting the prediction into readable format (as binary or thresholded outputs)
            prediction = (prediction > 0.5).astype(int)  # Threshold at 0.5 for binary classification
            predictions = {label: int(prediction[0][i]) for i, label in enumerate(labels)}

            st.write('Bidirectional LSTM Predictions:')
            st.json(predictions)
        else:
            st.write("Please enter a comment to classify.")
