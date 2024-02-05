from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROMPTS_DIR = ROOT_DIR.joinpath("prompts")
OUTPUT_DIR = ROOT_DIR.joinpath("output")
DATA_DIR = ROOT_DIR.joinpath("data")
EMAILS_DIR = DATA_DIR.joinpath('emails')
QUESTIONS_FILE_PATH = DATA_DIR.joinpath("questions.txt")
EMAIL_BODY_FIELD = 'email_body'
EMAIL_SENDER_NAME_FIELD = 'sender_name'
EMAIL_SUBJECT_FIELD = 'email_subject'
CACHE_DIR = ROOT_DIR.joinpath('cache')
