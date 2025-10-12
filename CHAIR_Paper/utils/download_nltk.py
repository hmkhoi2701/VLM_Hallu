import nltk

custom_path = '/home/user/khoihm/llava15/nltk_data'
nltk.data.path.append(custom_path)

nltk.download('wordnet', download_dir=custom_path)
nltk.download('punkt_tab', download_dir=custom_path)
nltk.download('omw-1.4', download_dir=custom_path)

from nltk.stem import WordNetLemmatizer
from pattern.text.en import singularize