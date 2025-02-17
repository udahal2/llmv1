import threading
import time
import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from colorama import Fore, Back, Style

# Function to print project and author info with colored output
def print_project_info():
    print(Fore.GREEN + Style.BRIGHT + "Project Name: Brendan Bycroft's Forked Transformer Visualization Project")
    print(Fore.CYAN + Style.BRIGHT + "Author: Brendan Bycroft")
    print(Fore.YELLOW + "This repository focuses on transformer models and their visualization.")
    print(Fore.MAGENTA + "Repository Link: [Your repository link here]")
    print(Fore.WHITE + "Description: A visualization tool to help understand transformer models.")
    print(Style.RESET_ALL)  # Reset to default color

# Function to simulate the visualization of a basic transformer model
def visualize_transformer():
    print(Fore.BLUE + Style.BRIGHT + "\nStarting Transformer Visualization...\n")

    # Example: Visualizing a simple Transformer model (BERT in this case)
    # Load BERT pre-trained model and tokenizer for a basic transformer demonstration.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Sample input text for transformer
    input_text = "Transformers are a powerful tool for natural language processing."

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Visualizing the model's attention mechanism (basic concept of attention heatmap)
    attention = outputs.attentions if 'attentions' in outputs.keys() else []

    if attention:
        print("Visualizing the Attention Mechanism...")

        # Choose a specific layer's attention weights to visualize
        attention_layer = attention[0][0]  # 1st layer's attention matrix
        attention_weights = attention_layer.detach().numpy()

        # Plotting the attention weights
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attention_weights, cmap='coolwarm')
        fig.colorbar(cax)
        plt.title("Attention Weights Visualization for Transformer Model (BERT)")
        plt.xlabel("Token Position")
        plt.ylabel("Token Position")
        plt.show()
    else:
        print("No attention weights available for visualization. Ensure the model supports it.")

    print(Style.RESET_ALL)  # Reset to default color

# Thread 1: Print project info and author info
thread1 = threading.Thread(target=print_project_info)

# Thread 2: Visualize the transformer model after a delay
def delayed_visualization():
    time.sleep(20)  # Simulate 20 seconds delay to clear Python engine memory
    visualize_transformer()

thread2 = threading.Thread(target=delayed_visualization)

# Starting the threads
thread1.start()
thread1.join()  # Wait for the first thread to finish before starting the second

thread2.start()  # Start the second thread after the first finishes

