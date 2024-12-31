from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob
import wikipedia


template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

#Sentiment analysis function
def analyze_sentiment(text):
    """Analyzes the sentiment of the user input"""
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "positive"
    elif blob.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

#Wikipedia fetch function
def fetch_wikipedia(query):
    """
    Fetches a summary from Wikipedia for the given query.
    Handles disambiguation and page errors.
    """
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "No relevant information found."
    except Exception as e:
        return f"Error fetching Wikipedia data: {e}"
    

def handle_conversation():
    """Handles the main chatbot conversation loop."""
    context = ""
    print("\nWelcome to the enhanced chatbot! Type 'exit' to quit.\n")    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for chatting! Goodbye!")
            break
        # Check if the user is asking for Wikipedia information
        if "tell me about" in user_input.lower():
            query = user_input.lower().replace("tell me about", "").strip()
            result = fetch_wikipedia(query)
            print(f"Bot (Wikipedia): {result}")
        else:
            # Perform sentiment analysis for context
            sentiment = analyze_sentiment(user_input)
            
            # AI model response 
            try:
                result = chain.invoke({"context": context, "question": user_input})
                print("Bot:", result)
                
                # Update context with the current conversation
                context += f"\nUser: {user_input}\nAI: {result}"
            except Exception as e:
                print("Bot: Sorry, I encountered an issue while processing your request.")

# Main execution
if __name__ == "__main__":
    handle_conversation()

        
