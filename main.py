import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Ask any question about Julien!")

# Adding a brief description or instruction
st.markdown("""
This project add embiddings to an open source LLM to answer any question about Julien GODFROY
""")

with st.sidebar:
    st.markdown("""
                # Julien's GPT4ALL
#### Model : mistral-7b-openorca.Q4_0.gguf
#### GitHub repo : https://github.com/jugodfroy/GPT4ALL-langchain-embeddings
""")
    st.write("## Ask Your Question Here")
    
    with st.form(key='my_form'):
        query = st.text_area(
            "Type your question in the box below:",
            max_chars=50,
            key="query",
            help="Keep your question clear and concise for the best results.",
            placeholder="Why should I hire Julien ?"
        )
        
        submit_button = st.form_submit_button('Submit')

st.subheader("Answer:")
if query:
    response = lch.query(query)
    print(response)
    st.text(textwrap.fill(response, width=85))
else:
    default_response = "You should hire Julien because he has a strong background in management and IT consulting. He demonstrated his ability to lead teams effectively by managing a 40-member board at Junior ISEP, Paris, where he achieved significant sales growth of over 200k euros. Additionally, Julien showcased his technical skills as an IT Consultant, developing a dynamic website for an architecture firm and conducting comprehensive interviews with key personnel to gain insights into operational workflows and technological requirements. His experience in both management and IT consulting makes him a valuable asset to any team."
    st.text(textwrap.fill(default_response, width=85))
