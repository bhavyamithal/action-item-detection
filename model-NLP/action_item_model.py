import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging


def load_csv_file():
    """
    Asks the user for the CSV file path and reads the data from the CSV file.
    Returns the DataFrame containing the data.
    """
    csv_file_path = input("Enter the path to the CSV file: ")
    data = pd.read_csv(csv_file_path, header=None, names=["start_time", "end_time", "speaker", "text"])
    return data


def action_item_classifier(data):
    """
    Classifies each sentence in the data as an action item or not.
    Returns a DataFrame with the action items.
    """
    model_name = "knkarthick/Action_Items"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1).tolist()

    data["action_label"] = ["Action Item" if pred == 1 else "No Action Item" for pred in predictions]
    action_items_df = data[data["action_label"] == "Action Item"][["text", "start_time"]]
    return action_items_df


def iterative_summarization(text, tokenizer, model):
    """
    Applies iterative summarization to the input text using the T5 model.
    Returns the summarized text.
    """
    num_iterations = 3
    summary = text

    for _ in range(num_iterations):
        inputs = tokenizer.encode("summarize: " + summary, max_length=512, truncation=True, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary


def text_summarizer(output_list):
    """
    Summarizes the transcript text using the T5 model with iterative summarization.
    Returns a DataFrame with the summarized action items.
    """
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    action_items = []
    for item in output_list:
        text = item["text"]
        summarized_text = iterative_summarization(text, tokenizer, model)
        action_items.append({"text": summarized_text, "ts": item["ts"]})

    action_items_df = pd.DataFrame(action_items)
    return action_items_df


def find_assignees(action_items_df):
    """
    Finds the assignee for each action item.
    Modifies the input DataFrame to include the 'assignee' column.
    """
    nlp = spacy.load("en_core_web_sm")

    def extract_first_word(text):
        words = text.split()
        if words:
            first_word = words[0].rstrip(',')
            return first_word
        else:
            return ""

    def classify_name(word):
        doc = nlp(word)
        if doc.ents and doc.ents[0].label_ == "PERSON":
            return word
        else:
            return "UNKNOWN"

    action_items_df['first_word'] = action_items_df['text'].apply(extract_first_word)
    action_items_df['assignee'] = action_items_df['first_word'].apply(classify_name)
    action_items_df = action_items_df.drop('first_word', axis=1)
    return action_items_df


def extract_action_items(action_items_df):
    """
    Extracts action items from the transcript text.
    Modifies the input DataFrame to include the 'action_items' column.
    """
    model_name = "sohamchougule/t5-base-finetuned-aeslc-test"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    action_items = []
    for item in action_items_df['text']:
        text = item
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        action_item = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, temperature=0.7)[0]
        decoded_action_item = tokenizer.decode(action_item, skip_special_tokens=True)
        action_items.append(decoded_action_item)

    action_items_df['action_items'] = action_items
    action_items_list = action_items_df[action_items_df['action_items'] != ''][['action_items', 'assignee', 'ts']].to_dict(orient='records')
    return action_items_list


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load CSV file
    data = load_csv_file()

    # Action Item Classifier
    action_items_df = action_item_classifier(data)
    logging.info("Action item classification completed.")

    # Text Summarizer
    output_list = action_items_df.rename(columns={"text": "text", "start_time": "ts"}).to_dict(orient="records")
    action_items_df = text_summarizer(output_list)
    logging.info("Text summarization completed.")

    # Find Assignees
    action_items_df = find_assignees(action_items_df)
    logging.info("Assignee finding completed.")

    # Extract Action Items
    action_items_list = extract_action_items(action_items_df)
    logging.info("Action item extraction completed.")

    # Print each item in the list
    for item in action_items_list:
        print(item)


if __name__ == "__main__":
    main()