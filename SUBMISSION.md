## This guide provides instructions on how to set up the codebase and run the pipeline for Meeting Action Item Detection.

# Setup
- Clone the repository:
```git clone https://github.com/DiveHQ-Octernships/dive-ml-engineering-octernship-bhavyamithal```

- Navigate to the project directory:
```cd <project_directory>```

- Install the required dependencies. Make sure you have Python and pip installed. Run the following command:
```pip install -r requirements.txt```

# Usage
Prepare your CSV file containing the meeting transcript data.

- Run the code:
```python model-NLP/action_item_model.py```
You will be prompted to enter the path to the CSV file. Provide the full path to the file and press Enter.

The code will process the transcript data, perform action item detection, and display the action items found in the transcript.

Review the action items printed in the console.

Please ensure that your CSV file is correctly formatted and contains the required columns: **"start_time", "end_time", "speaker", and "text".**

# Limitations
The code assumes that the CSV file provided contains the required columns in the specified format.

# Conclusion
Following the instructions in this guide, you can set up the codebase and run the pipeline to detect action items in meeting transcripts. If you encounter any difficulties or have further questions, please don't hesitate to reach out for assistance.

*Happy meeting action item identification!*
