import json

from dotenv import load_dotenv

from timestamp_types import MmsLanguage, Translation

model_name = "ctc_alignment_mling_uroman_model.pt"
model_url = (
    "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
)
dict_name = "ctc_alignment_mling_uroman_model.dict"
dict_url = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt"

load_dotenv()

mms_languages: list[MmsLanguage] = json.load(
    open("data/mms_languages.json", encoding="utf-8")
)
