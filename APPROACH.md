# Task 1: Training Data

## Initial Approach:
I began by researching various action item detection datasets and found three main options: **AMI, ICSI, and AMC-A**. Among these, I decided to use the AMI corpus due to its larger size and availability in a processed JSON format.

## Approaches Considered:
After exploring the AMI corpus, I discovered that it contains 100 meetings with audio and transcriptions. The SpokenNLP team had already processed the corpus and provided it in JSON format, which I utilized for training. This saved time on data processing.

## Prepraing the Data:
During the data exploration phase, I noticed that the SpokenNLP team had used a pre-trained model (StructBERT) fine-tuned on their Chinese corpus for action item classification. However, since I needed an English action item classification model, I opted for a pre-trained sequence classification model.

**Note:**
I have not retrained the pre-trained models used in this project due to time constraints. The models are already trained on large datasets and have achieved state-of-the-art performance. However, I have included the code for parsing the AMI corpus XML files in processing-AMI/ami_data_process.py, which can be used for future work.

## Future Work:
In the research paper *"Meeting Action Item Detection with Regularized Context Modeling"* by Jiaqing Liu, Chong Deng, Qinglin Zhang, Qian Chen, Wen Wang, the authors proposed incorporating local and global context for action item classification. This approach could be explored further in the future.


# Task 2: Develop NLP Model

## Approaches Considered:
1. **Action Item Classification:** The first step is to classify each sentence in the transcript as an action item or not. I considered using a pre-trained sequence classification model for this task.
2. **Text Summarization:** To provide a concise summary of the transcript, I explored using a pre-trained T5 model for iterative summarization.
3. **Finding Assignees:** Next, I aimed to identify the assignee for each action item. I utilized a pre-trained name classifier model (spacy "en_core_web_sm") to extract potential assignees.
4. **Action Item Extraction:** Lastly, I employed a pre-trained T5 model specifically fine-tuned for action item extraction to generate the final action items from the transcript.
5. Also considered using ```rasa``` but decided against it due to time constraints.
6. Read about people using ```roberta-base-action-item-classification```, but it wasn't available publically.

## Assumptions Made:
- The input transcript is in a structured format with columns **"start_time", "end_time", "speaker", and "text"**.
- The action items in the transcript are explicitly mentioned as sentences and can be identified based on their context from the same sentence only(and not local or global context).
- The assignee of an action item is the first word in the sentence, assuming it represents a person's name.

## Approach Selection:
- For action item classification, I chose a pre-trained sequence classification model as it has shown promising results in classifying text sequences.
- For text summarization, I opted for a pre-trained T5 model as it has been widely used for generating summaries and has achieved state-of-the-art performance.
- For finding assignees, I selected the spacy "en_core_web_sm" model for its ability to extract named entities and classify them as person names.
- For action item extraction, I utilized a fine-tuned T5 model specifically trained for the task, which has demonstrated effectiveness in generating action items from text.

## Approaches Discarded:
- I considered using LSTM models for action item classification but decided against it due to lower accuracy compared to the pre-trained sequence classification models.
- I explored other text summarization models such as BART and GPT but chose T5 as it provides good summarization capabilities and is widely adopted.
- I evaluated different models for named entity recognition, including deep learning-based approaches, but selected the spacy "en_core_web_sm" model for its simplicity and reasonable performance.

## Limitations:
- The accuracy of the action item classification model depends on the quality of the model's pre-training.
- The text summarization model might not capture all the important details in the transcript, leading to potential loss of information. But it was necessary to provide a concise summary of the transcript as longer transcripts would be difficult to read and process.
- The assignee identification relies on the assumption that the assignee's name appears as the first word in the action item sentence, which may not always hold true.
- The action item extraction model's performance may vary depending on the quality and diversity of the pre-training data.

## Future Work:
Explore other pre-trained models for action item classification, text summarization, and action item extraction.
Train the pre-trained models on the AMI corpus and evaluate their performance.

## Citations:
- Action Item Classification model: [knkarthick/Action_Items](https://huggingface.co/knkarthick/Action_Items)
- Text Summarization model: [t5-base](https://huggingface.co/t5-base)
- Name Classifier model: [spacy "en_core_web_sm"](https://spacy.io/models/en#en_core_web_sm)
- Action Item Extraction model: [sohamchougule/t5-base-finetuned-aeslc-test](https://huggingface.co/sohamchougule/t5-base-finetuned-aeslc-test)
- "Meeting Action Item Detection with Regularized Context Modeling" by Jiaqing Liu, Chong Deng, Qinglin Zhang, Qian Chen, Wen Wang. [Link to the paper](https://arxiv.org/abs/2101.00109)
- SpokenNLP/action-item-detection: [Link to the repository](https://github.com/alibaba-damo-academy/SpokenNLP/tree/main/action-item-detection)

# Aknowledgements:
I would like to express my gratitude to the Hugging Face team for providing the pre-trained models and the SpokenNLP team for their valuable research and codebase on action item detection. I also borrowed and modified code from the SpokenNLP repository for the data processing of the AMI corpus.
