import os
import logging
import threading
import time
import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from colorama import Fore, Style

# Configure logging to log to 'logs/visualization.log'
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "visualization.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Log start of the script
logging.info("Script execution started.")

# Function to print project and author info with colored output
def print_project_info():
    try:
        print(Fore.GREEN + Style.BRIGHT + "Project Name: Brendan Bycroft's Forked Transformer Visualization Project")
        print(Fore.CYAN + Style.BRIGHT + "Author: Brendan Bycroft")
        print(Fore.YELLOW + "This repository focuses on transformer models and their visualization.")
        print(Fore.MAGENTA + "Repository Link: [Your repository link here]")
        print(Fore.WHITE + "Description: A visualization tool to help understand transformer models.")
        print(Style.RESET_ALL)  # Reset to default color

        # Log project information
        logging.info("Printed project info and author details.")
    except Exception as e:
        logging.error(f"Error printing project info: {e}")
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

# Function to visualize the transformer model (BERT)
def visualize_transformer():
    try:
        print(Fore.BLUE + Style.BRIGHT + "\nStarting Transformer Visualization...\n")

        # Load BERT pre-trained model and tokenizer for a basic transformer demonstration
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
            logging.info("Visualizing the Attention Mechanism...")

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
            logging.warning("No attention weights available for visualization.")

        print(Style.RESET_ALL)  # Reset to default color
        logging.info("Completed transformer visualization.")
    except Exception as e:
        logging.error(f"Error visualizing transformer: {e}")
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

# Thread 1: Print project info and author info
def start_printing_info():
    print_project_info()

# Thread 2: Visualize the transformer model after a delay
def delayed_visualization():
    try:
        time.sleep(20)  # Simulate 20 seconds delay to clear Python engine memory
        visualize_transformer()
    except Exception as e:
        logging.error(f"Error in delayed visualization: {e}")
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

# Main function to start threads
def main():
    try:
        logging.info("Starting the threads for project info and transformer visualization.")

        # Thread 1: Print project info
        thread1 = threading.Thread(target=start_printing_info)

        # Thread 2: Visualize transformer after delay
        thread2 = threading.Thread(target=delayed_visualization)

        # Start thread 1
        thread1.start()
        thread1.join()  # Wait for the first thread to finish before starting the second

        # Start thread 2
        thread2.start()
        thread2.join()  # Wait for the second thread to finish

        logging.info("Script execution completed successfully.")

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

# Run the main function
if __name__ == "__main__":
    main()
