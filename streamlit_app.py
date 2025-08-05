import os
import streamlit as st
from agent import create_agent

# Title of the web app
st.title("LangChain Code Agent")

# Sidebar for API key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Radio button to choose between generating or explaining code
task_type = st.radio("Select task type", ("Write code", "Explain code"))

# Input fields for language and prompt
language = st.text_input("Programming language (e.g. python, javascript, c++)")

# Depending on the task type, show appropriate prompt field
if task_type == "Write code":
    user_input = st.text_area("Describe the programming task you want the agent to perform")
else:
    user_input = st.text_area("Paste the code you want explained")

# Submit button
if st.button("Submit"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not language or not user_input:
        st.warning("Please provide both the programming language and the task/code.")
    else:
        # Set the API key for the OpenAI client
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Create the agent
        agent = create_agent()

        # Build the prompt based on the selected task type
        if task_type == "Write code":
            prompt = f"Write {language} code to: {user_input}"
        else:
            prompt = f"Explain the following {language} code:\n{user_input}"

        # Generate and display the response
        with st.spinner("Processing..."):
            try:
                result = agent.run(prompt)
                # Display code in a code block for write tasks, otherwise show as plain text
                if task_type == "Write code":
                    st.code(result, language=language)
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"Error invoking the agent: {e}")
