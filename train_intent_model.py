# train_intent_model.py

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Step 1: Load Your Dataset ===
df = pd.read_csv("command_dataset.csv")  # Make sure this file exists in the same folder
X = df["command"].astype(str)
y = df["intent"].astype(str)

# === Step 2: Load Sentence Embedding Model ===
print("[+] Encoding commands using sentence-transformers...")
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
X_embeddings = embed_model.encode(X.tolist(), show_progress_bar=True)

# === Step 3: Split and Train ===
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === Step 4: Evaluate Model ===
print("\n[+] Evaluation Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# === Step 5: Save Models ===
joblib.dump(clf, "intent_model_LR.pkl")
joblib.dump(embed_model, "embedding_model.pkl")
print("\n[+] Models saved: intent_model_LR.pkl & embedding_model.pkl")

"""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()"""