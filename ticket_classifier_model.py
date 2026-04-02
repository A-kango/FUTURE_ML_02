import pandas as pd
import numpy as np
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
FILE_PATH = r'C:\Users\Sober\OneDrive\Desktop\Ticket ML\complaints.csv'
BATCH_SIZE = 15000
TOTAL_ROWS = 1000000
TOP_CATS = ['Credit Reporting', 'Cards', 'Banking', 'Loans', 'Mortgage', 'Debt Collection']

category_map = {
    'Credit reporting, credit repair services, or other personal consumer reports': 'Credit Reporting',
    'Credit reporting or other personal consumer reports': 'Credit Reporting',
    'Credit card or prepaid card': 'Cards', 'Credit card': 'Cards', 'Prepaid card': 'Cards',
    'Checking or savings account': 'Banking', 'Bank account or service': 'Banking',
    'Vehicle loan or lease': 'Loans', 'Student loan': 'Loans', 'Consumer Loan': 'Loans',
    'Payday loan, title loan, or personal loan': 'Loans',
    'Mortgage': 'Mortgage', 'Debt collection': 'Debt Collection'
}

# --- 2. TOOLS FOR SCALE ---
# HashingVectorizer uses very little memory compared to Tfidf
vectorizer = HashingVectorizer(n_features=2**18, stop_words='english', alternate_sign=False)
# SGDClassifier supports incremental training (partial_fit)
model = SGDClassifier(loss='log_loss', random_state=42)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'x{2,}', '', text) # Remove XXXX redactions
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- 3. BATCH TRAINING LOOP ---
print(f"Starting Batch Training: {TOTAL_ROWS} rows in chunks of {BATCH_SIZE}...")

# Stream the CSV instead of loading it all
reader = pd.read_csv(FILE_PATH, chunksize=BATCH_SIZE, low_memory=False)

rows_processed = 0
for chunk in reader:
    if rows_processed >= TOTAL_ROWS:
        break
    
    # Cleaning and Mapping
    df_batch = chunk[['narrative', 'Product']].dropna()
    df_batch['Product'] = df_batch['Product'].replace(category_map)
    df_batch = df_batch[df_batch['Product'].isin(TOP_CATS)]
    
    if df_batch.empty:
        continue
    
    # Feature Extraction
    X_batch = vectorizer.transform(df_batch['narrative'].apply(clean_text))
    y_batch = df_batch['Product']
    
    # partial_fit updates the model weights incrementally
    model.partial_fit(X_batch, y_batch, classes=TOP_CATS)
    
    rows_processed += BATCH_SIZE
    print(f"Status: {rows_processed}/{TOTAL_ROWS} rows trained.")

# --- 4. EVALUATION ---
print("\nEvaluating on a fresh test batch...")
# Load a specific small chunk for testing that wasn't in the training set
test_df = pd.read_csv(FILE_PATH, skiprows=TOTAL_ROWS, nrows=15000, 
                     names=chunk.columns, low_memory=False)
test_df = test_df[['narrative', 'Product']].dropna()
test_df['Product'] = test_df['Product'].replace(category_map)
test_df = test_df[test_df['Product'].isin(TOP_CATS)]

X_test = vectorizer.transform(test_df['narrative'].apply(clean_text))
y_test = test_df['Product']
y_pred = model.predict(X_test)

print("\n--- Optimized Performance (300k Samples) ---")
print(classification_report(y_test, y_pred))

# --- 5. VISUALIZATION ---
cm = confusion_matrix(y_test, y_pred, labels=TOP_CATS)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=TOP_CATS, yticklabels=TOP_CATS, cmap='Greens')
plt.title('Confusion Matrix: 300k Training Samples')
plt.show()

# --- 6. PRIORITY LOGIC & DEMO ---
def assign_priority(product):
    return 'High' if product in ['Debt Collection', 'Mortgage'] else \
           'Medium' if product in ['Credit Reporting', 'Cards'] else 'Low'

def classify_new_ticket(text):
    vec = vectorizer.transform([clean_text(text)])
    cat = model.predict(vec)[0]
    return f"Category: {cat} | Priority: {assign_priority(cat)}"

print("\n--- System Test ---")
print(classify_new_ticket("I am being harassed by collectors for a debt I already paid!"))










