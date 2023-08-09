import os
import openai
import gradio as gr
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper,GPTListIndex
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def read_json_files(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), "r") as file:
                json_data = file.read()
                data.append(json_data)
    return data

def construct_index(data):
    # Parameters for index construction
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0.2
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        max_tokens=num_outputs
    ))

    # Construct and save the index
    index = GPTVectorStoreIndex(data, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    # Load the index from disk
    index = GPTVectorStoreIndex.from_documents('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

# Specify the path to the directory containing JSON files
data_directory = "/home/cocodernero/Downloads/dataset.json"

# Read data from JSON files in the directory
data = read_json_files(data_directory)

# Uncomment the next line if you want to construct the index before launching the interface
# index = construct_index(data)

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
    outputs=gr.outputs.Textbox(label="Chatbot Response"),
    title="Custom-trained AI Chatbot"
)

# Launch the Gradio interface
iface.launch(share=True)

