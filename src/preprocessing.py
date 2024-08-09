from string import punctuation
import re
import json
import pandas as pd
from py_vncorenlp import VnCoreNLP

# Initialize the VnCoreNLP
class RDRSegmenter:
    def __init__(self, save_dir):
        # Khởi tạo VnCoreNLP với các annotators được yêu cầu
        self.rdrsegmenter = VnCoreNLP(annotators=["wseg"], save_dir=save_dir)

    def word_segment(self, text):
        return self.rdrsegmenter.word_segment(text)

def preprocessing_paragraph(paragraph: str, wseg = False) -> str:
    # Remove specified punctuation characters from the punctuation string
    modified_punctuation = ''.join(char for char in punctuation if char not in '"\').?}]_')

    # Split the paragraph into sentences
    sentences = paragraph.strip().split('.')
    sentences = sum([para.split(';') for para in sentences], [])

    processed_sentences = []
    for sentence in sentences:
        try: 
            sentence = (sentence.strip() + '.') if (sentence[-1] not in ['.', '?'] and sentence[-1] not in modified_punctuation)\
            else (sentence[:-1].strip() + '.') if (sentence[-1] in modified_punctuation)\
            else sentence.strip()
            if sentence.strip():
                processed_sentences.append(re.sub(r"\s+", " ", sentence))
        except:
            pass



    # Join the processed sentences to form the final result
    if wseg:
        return ' '.join(processed_sentences)
    else:
        return ' '.join(processed_sentences).replace('_', ' ')


def load_error_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data = []
            for json_str in content.split('\n'):  # Assuming each JSON object is on a new line
                if json_str.strip():  # Ensure the line is not empty
                    data.append(json.loads(json_str))
        print("JSON data successfully loaded.")
        # You can now work with the `data` variable
        print(data)  # Print the loaded data to verify
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data




def covert_csv(df):
    # Reshape the DataFrame
    
    print(len(df))
    reshaped_data = []

    for index, row in df.iterrows():

        try:
            _context = row['context'], row['evidence_A_other'], row['evidence_B_other'], row['evidence_C_other'], row['evidence_D_other']
        except:
            _context = row['context']
        
        for option in ['A', 'B', 'C', 'D']:
            reshaped_data.append({
                
                'wseg_context': _context,
                'ques_opt': row['question'] + row[f'{option}'],
                'label': 1 if option in row['result'] else 0,
            })

    reshaped_df = pd.DataFrame(reshaped_data)

    print(len(reshaped_df))


def covert_csv_1(df):
    # Reshape the DataFrame
    
    print(len(df))
    reshaped_data = []

    for index, row in df.iterrows():

        try:
            _context = ' '.join(row['context'], row['evidence_A_other'], row['evidence_B_other'], row['evidence_C_other'], row['evidence_D_other'])
        except:
            _context = row['context']
            
        reshaped_data.append(
            {
            'context': _context,
            'ques_opt': row['question'] + ' '.join([row[f'{option}'] for option  in ['A', 'B', 'C', 'D']]),
            'label': [1 if option in row['result'] else 0 for option  in ['A', 'B', 'C', 'D'] ],
        }
        )
        
    reshaped_df = pd.DataFrame(reshaped_data)

    print(len(reshaped_df))
    
    return reshaped_df
