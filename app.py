import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Page configuration
st.set_page_config(
    page_title="Roman Urdu–English Code-Switching NLP",
    page_icon="🔤",
    layout="centered"
)

# Hugging Face model
MODEL_ID = "emanfatimaa05/code-switching-codesaviours-si26-eman"


# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# Title
st.title("🔤 Roman Urdu–English Code-Switching NLP")

st.write(
    "Enter a Roman Urdu–English mixed sentence and the model "
    "will identify each word as Urdu (URD) or English (ENG)."
)


# User input
text = st.text_input(
    "Enter a sentence:",
    placeholder="Aaj mera mood nahi hai for anything"
)


# Prediction
if st.button("Analyze Sentence"):

    if not text.strip():
        st.warning("Please enter a sentence first.")

    else:
        words = text.split()

        # Tokenize
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)[0]

        # Map predictions back to words
        word_ids = inputs.word_ids(batch_index=0)

        results = []
        previous_word_id = None

        for token_index, word_id in enumerate(word_ids):

            if word_id is None:
                continue

            if word_id != previous_word_id:

                word = words[word_id]
                label = model.config.id2label[
                    predictions[token_index].item()
                ]

                results.append((word, label))

            previous_word_id = word_id

        # Display results
        st.subheader("Language Identification")

        for word, label in results:

            if label == "URD":
                st.write(f"**{word}** → 🇵🇰 URD")

            else:
                st.write(f"**{word}** → 🇬🇧 ENG")

        # Summary
        urd_count = sum(1 for _, label in results if label == "URD")
        eng_count = sum(1 for _, label in results if label == "ENG")

        st.subheader("Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Urdu Words", urd_count)

        with col2:
            st.metric("English Words", eng_count)
