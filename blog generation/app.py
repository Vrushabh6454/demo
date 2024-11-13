import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


# Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # Initialize LLama2 model
        llm = CTransformers(
            model='models\llama-2-7b-chat.Q8_0.gguf',
            model_type='llama',
            config={'max_new_tokens': 256, 'temperature': 0.01}
        )

        # Prompt template
        template = """
        Write a blog for {blog_style} job profile for the topic "{input_text}" within {no_words} words.
        """

        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template
        )

        # Generate the response from the LLama 2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        return response

    except Exception as e:
        return f"Error: {e}"

# Streamlit app setup
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered')

st.header("Generate Blogs 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

# Columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    if input_text and no_words.isdigit():  # Validate input
        with st.spinner('Generating blog...'):
            response = getLLamaresponse(input_text, no_words, blog_style)
        st.write("### Generated Blog:")
        st.write(response)
    else:
        st.error("Please provide valid inputs for the topic and word count.")
