import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://models.inference.ai.azure.com"
model_name = "meta-llama-3.1-405b-instruct"
api_token = os.getenv("token")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_token),
)

def get_ai_answer(client, model_name, question, temperature=1.0, top_p=1.0, max_tokens=1000):
    """
    Queries the AI model for an answer to a user's question.

    Args:
        client (object): The AI client object.
        model_name (str): The name of the AI model.
        question (str): The user's question.
        temperature (float, optional): The temperature parameter. Defaults to 1.0.
        top_p (float, optional): The top_p parameter. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 1000.

    Returns:
        str: The AI's answer.
    """
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=question),
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model=model_name
    )
    return response


# Example usage:
question = "What is the capital of Uganda?"
answer = get_ai_answer(client, model_name, question)
print(answer)

def chat_with_model():
    print("Chatbot, Hello how can I assist you today?")
    conversation = [
        SystemMessage(content="You are helpfull assistant")
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye")
            break

        conversation.append(UserMessage(content=user_input))

        response = get_ai_answer(client, model_name, user_input)

        chatbot_message = response.choices[0].message.content
        print("Chatbot:", chatbot_message)
        conversation.append(SystemMessage(content=chatbot_message))

chat_with_model()