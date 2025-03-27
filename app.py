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
        return str(e)  # return error message as string


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
                    st.error(f"Oops! One or more words are not in the vocabulary. Please try different words.\n\nDetails: {answer}")
                else:
                    st.subheader("Top 10 Similar Words")
                    for word, score in answer:
                        st.write(f"**{word}** — similarity score: `{score:.4f}`")

            else:
                st.warning("Please enter all three words.")


if __name__ == "__main__":
    main()
