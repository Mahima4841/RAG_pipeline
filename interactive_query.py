import sys
from query_rag import query_rag, print_wrapped, generate_response

def interactive_query():
    """
    Interactive query interface for the RAG system
    """
    print("RAG Interactive Query System")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop")
    print()
    
    while True:
        try:
            # Get user query
            query = input("Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            print("\n" + "="*50)
            print(f"Query: {query}")
            print("="*50)
            
            # Query the RAG system
            results = query_rag(query, top_k=1)
            
            if results is None:
                print("Failed to load embeddings. Please run generate_embeddings.py first.")
                continue
            
            top_results_dot_product, documents = results
            
            print(f"Results: ")
            
            for score, idx in zip(top_results_dot_product[0][0], top_results_dot_product[1][0]):
                print(f"Score: {score}")
                print("Text: ")
                context = print_wrapped(documents[idx])
                print("\n")
                
                # Generate response using Ollama
                response = generate_response(context, query)
                print("Generated Response:")
                print(response)
                print("\n" + "="*50 + "\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    interactive_query() 