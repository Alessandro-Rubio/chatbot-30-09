import ollama

def generate_response(user_message: str, model: str = "llama3.1") -> str:
    """
    Generate a response from Ollama LLM
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": user_message
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")