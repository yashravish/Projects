import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

class StudyAssistantChatbot:
    def __init__(self):
        self.system_prompt = (
            "You are a helpful and knowledgeable Study Assistant. Your tasks include summarizing information, "
            "explaining difficult topics, providing study strategies, creating study schedules, and giving motivational advice. "
            "Engage naturally with the user, ask clarifying questions if needed, and offer helpful suggestions."
        )
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def generate_response(self, user_input):
        # Add the user input to the conversation history
        self.chat_history.append({"role": "user", "content": user_input})

        # Send the API request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=self.chat_history,
            max_tokens=200,
            temperature=0.6,
            n=1
        )

        # Extract and save the chatbot's response
        chatbot_reply = response.choices[0].message["content"]
        self.chat_history.append({"role": "assistant", "content": chatbot_reply})
        
        return chatbot_reply

    def clear_history(self):
        """Resets the conversation history."""
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

# Chat loop for interaction
if __name__ == "__main__":
    chatbot = StudyAssistantChatbot()
    print("Study Assistant: Hi! I’m your study assistant. I can help with summaries, explanations, study plans, and more. Type 'reset' to start fresh or 'exit' to end the chat.")

    while True:
        # Get user input
        user_input = input("You: ")

        # End or reset conversation if needed
        if user_input.lower() == "exit":
            print("Study Assistant: Goodbye! Happy studying!")
            break
        elif user_input.lower() == "reset":
            chatbot.clear_history()
            print("Study Assistant: I've reset our conversation. How can I assist you now?")
            continue

        # Generate and print chatbot's response
        response = chatbot.generate_response(user_input)
        print(f"Study Assistant: {response}")
