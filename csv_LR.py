import pandas as pd
import random

# Load original dataset
file_path = "command_dataset.csv"
df = pd.read_csv(file_path)

# Group by intent and list existing commands
intent_groups = df.groupby("intent")["command"].apply(list).to_dict()

# Synonyms for common command parts (simple augmentation strategy)
synonyms = {
    "open": ["launch", "start", "access", "go to", "fire up"],
    "play": ["start", "stream", "put on"],
    "set": ["schedule", "create", "make"],
    "turn on": ["enable", "activate", "switch on"],
    "turn off": ["disable", "deactivate", "switch off"],
    "check": ["see", "look at", "review"],
    "find": ["locate", "search for", "look for"],
    "show": ["display", "bring up", "pull up"],
    "create": ["make", "add", "generate"],
    "resume": ["continue", "start again"],
    "pause": ["halt", "hold", "pause"],
    "stop": ["end", "terminate", "kill"],
}

# Function to generate variations using synonyms
def generate_variations(command, n=10):
    words = command.split()
    variants = set()
    for _ in range(n):
        new_words = []
        for word in words:
            key = word.lower()
            if key in synonyms and random.random() < 0.5:
                new_words.append(random.choice(synonyms[key]))
            else:
                new_words.append(word)
        variants.add(" ".join(new_words))
    return list(variants)

# Generate 10 more variations per intent
new_data = []
for intent, commands in intent_groups.items():
    base_command = commands[0]  # take the first one for expansion
    new_variants = generate_variations(base_command, 15)
    for new_cmd in new_variants:
        if new_cmd not in commands:
            new_data.append({"command": new_cmd, "intent": intent})

# Combine original and new data
augmented_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
augmented_df = augmented_df.drop_duplicates().sort_values(by=["intent", "command"])

# Save to a new CSV
augmented_path = "command_dataset.csv"
augmented_df.to_csv(augmented_path, index=False)
augmented_path
