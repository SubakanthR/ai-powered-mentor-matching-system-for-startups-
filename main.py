import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import re
import joblib

# Load the mentor dataset
def load_mentor_dataset(csv_path):
    return pd.read_csv(csv_path)

# Define the pre-trained model and tokenizer for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the possible domains
domains = [
    "Technology", "Health", "Finance", "Education", "Marketing", "Legal",
    "Real Estate", "Entertainment", "Transportation", "Energy",
    "Food and Beverage", "E-commerce", "Social Impact", "Agriculture",
    "Manufacturing", "Tourism", "Fashion"
]

# Function to identify the domain from the startup idea
def identify_domain(startup_idea):
    result = classifier(startup_idea, domains)
    return result['labels'], result['scores']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Function to prepare the dataset
def prepare_dataset(mentors):
    le = LabelEncoder()
    mentors['Domain'] = le.fit_transform(mentors['Domain'])
    mentors['Specializations'] = mentors['Specializations'].apply(clean_text)
    mentors['Industry-specific Knowledge'] = mentors['Industry-specific Knowledge'].apply(clean_text)
    return mentors, le

# Function to train the model
def train_model(mentors):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(mentors['Specializations'] + ' ' + mentors['Industry-specific Knowledge'])
    y = mentors['Domain']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model with hyperparameter tuning
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean CV score: {cv_scores.mean()}')

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {test_accuracy}')

    # Save the model, tfidf vectorizer, and label encoder
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    # Save the accuracy metrics
    with open('accuracy_metrics.txt', 'w') as f:
        f.write(f'Mean CV score: {cv_scores.mean()}\n')
        f.write(f'Test accuracy: {test_accuracy}\n')

    return best_model, tfidf, le

# Function to match mentee with mentors
def match_mentee_with_mentors(mentors, mentee_name, mentee_location, startup_idea, model, tfidf, le):
    identified_domains, confidence_scores = identify_domain(startup_idea)

    # Get the top 3 domains with the highest confidence scores
    top_domains = identified_domains[:3]
    top_scores = confidence_scores[:3]

    # Transform the input text to match the TF-IDF model
    input_text = [clean_text(startup_idea)]
    X_input = tfidf.transform(input_text)

    # Predict the domain
    predicted_domain = model.predict(X_input)
    predicted_domain_label = le.inverse_transform(predicted_domain)

    # Filter mentors based on the predicted domain and mentee's location
    matching_mentors = mentors[(mentors['Domain'] == predicted_domain[0]) & (mentors['Location'].str.lower() == mentee_location.lower())]

    print(f"\nMentee Name: {mentee_name}")
    print(f"Location: {mentee_location}")
    print(f"Startup Idea: {startup_idea}")
    print("\nTop 3 Identified Domains and Confidence Scores:")
    for domain, score in zip(top_domains, top_scores):
        print(f"{domain}: {score:.4f}")

    if matching_mentors.empty:
        print("\nNo matching mentors found.")
    else:
        print("\nMatching Mentors:")
        print(matching_mentors)

def main():
    # Load mentor dataset
    mentors = load_mentor_dataset('path/to/your/mentor_dataset.csv')

    # Prepare the dataset and train the model
    mentors, le = prepare_dataset(mentors)
    model, tfidf, le = train_model(mentors)

    while True:
        mentee_name = input("Enter mentee name (or type 'done' to finish): ")
        if mentee_name.lower() == 'done':
            break
        mentee_location = input("Enter mentee location: ")
        startup_idea = input("Enter mentee startup idea: ")

        match_mentee_with_mentors(mentors, mentee_name, mentee_location, startup_idea, model, tfidf, le)

if __name__ == "__main__":
    main()
