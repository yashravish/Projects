import spacy
from spacy.util import minibatch, compounding

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# For demonstration, we will add a text categorizer to classify intents (e.g., "balance inquiry", "card freeze")
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Add two labels for demonstration
textcat.add_label("BALANCE_INQUIRY")
textcat.add_label("CARD_FREEZE")

# Training data: list of tuples (text, {"cats": {label: value}})
train_data = [
    ("What is my account balance?", {"cats": {"BALANCE_INQUIRY": 1, "CARD_FREEZE": 0}}),
    ("I want to freeze my card immediately", {"cats": {"BALANCE_INQUIRY": 0, "CARD_FREEZE": 1}}),
    ("How much money do I have in my savings?", {"cats": {"BALANCE_INQUIRY": 1, "CARD_FREEZE": 0}}),
    ("Please block my debit card", {"cats": {"BALANCE_INQUIRY": 0, "CARD_FREEZE": 1}})
]

# Training loop
n_iter = 10
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        losses = {}
        batches = minibatch(train_data, size=compounding(2.0, 4.0, 1.2))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        print(f"Iteration {i + 1}, Losses: {losses}")

# Save the model to disk
nlp.to_disk("bankbuddy_nlp_model")
print("Model saved to 'bankbuddy_nlp_model'")
