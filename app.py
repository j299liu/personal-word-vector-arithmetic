"""
Prototype Steamlit app for word vector arithmetic.
The app imports vector_db from glove-wiki-gigaword-100, not word2vec-google-news-300 since it is too large to run for Streamlit.
"""
import streamlit as st
import gensim.downloader as api

@st.cache_resource
def load_vector_db():
    """Load a pre-trained word vectors. This model has 100-dimensional vectors for 1.2 million words and phrases."""
    vector_db = api.load('glove-wiki-gigaword-100')
    return vector_db


def word_vector_arithmetic(vector_db, pos1, pos2, neg1):
    try:
        results = vector_db.most_similar(positive=[pos1, pos2], negative=[neg1], topn=10)
        return results
    except KeyError as e:
        missing_word = str(e).split("'")[1] if "'" in str(e) else "a word"
        return missing_word  # return error message as string


def main():
    """Main function for the Streamlit app"""
    st.title("Wikipedia and Gigaword Vector Arithmetic Explorer")
    st.write("Using pre-trained glove-wiki-gigaword-100 model.")
    st.write(
        "Enter two positive words and one negative word, and we'll show words similar to the result of: **Positive word 1 - Negative word + Positive word 2**")

    with st.form("word_vector_form"):
        pos1 = st.text_input("Positive word 1")
        neg1 = st.text_input("Negative word")
        pos2 = st.text_input("Positive word 2")
        submitted = st.form_submit_button("Compute Similarity")

        if submitted:
            if pos1 and pos2 and neg1:
                with st.spinner("Loading word vectors..."):
                    vector_db = load_vector_db()
                with st.spinner("Calculating similarity..."):
                    answer = word_vector_arithmetic(vector_db, pos1, pos2, neg1)

                if isinstance(answer, str):
                    st.write(f"Oops! The word '{answer}' is not in the database. Please change to different words.\n\nNote: The word vector used in this demo, based on the GloVe Wiki-Gigaword-100 model, was trained on only 1.2 million words from the 2014 Wikipedia dump and Gigaword 5, so it does not cover all possible English vocabulary.")
                else:
                    st.subheader("Top 10 Similar Words")
                    for word, score in answer:
                        st.write(f"**{word}** — similarity score: `{score:.4f}`")

            else:
                st.warning("Please enter all three words.")


if __name__ == "__main__":
    main()
