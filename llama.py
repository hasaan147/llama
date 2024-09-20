import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "mattshumer/Reflection-Llama-3.1-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the Streamlit app layout
st.title("LLaMA AI Chatbot")
st.write("Ask me anything about AI!")

# Input text box for user to type their question
user_input = st.text_input("Your Question:", "")

# Button to generate response
if st.button("Generate Response"):
    if user_input:
        # Tokenize the input text
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Generate response
        outputs = model.generate(inputs["input_ids"], max_length=100)
        
        # Decode the generated response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the response
        st.write("**Response:**")
        st.write(generated_text)
    else:
        st.warning("Please enter a question!")
