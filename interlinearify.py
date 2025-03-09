import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from model import load_model
from utils import align_matches
from timestamp_types import File
import json

# Load MMS languages for validation (if needed)
mms_languages = json.load(open("mms_languages.json"))

# Load the alignment model and dictionary
model, dictionary = load_model()

def validate_language(language):
    """Check if the language is supported for alignment."""
    return any(lang["iso"] == language and lang["align"] for lang in mms_languages)

def process_pair(audio_path, text_path, language, separator='lineBreak'):
    """Process a single audio-text pair to get aligned timestamps."""
    audio_filename = Path(audio_path).name
    text_filename = Path(text_path).name
    directory = str(Path(audio_path).parent)
    
    files = [
        (audio_filename, audio_path),
        (text_filename, text_path)
    ]
    matched_files = [(files[0], files[1])]  # Manually pair audio and text
    
    if not validate_language(language):
        raise ValueError(f"Unsupported language: {language}")
    
    return align_matches(directory, language, separator, matched_files, model, dictionary)

def combine_audio(audio1_path, timestamps1, audio2_path, timestamps2):
    """Combine audio segments from both pairs into an interlinear audio file."""
    audio1 = AudioSegment.from_file(audio1_path)
    audio2 = AudioSegment.from_file(audio2_path)
    
    sections1 = sorted(timestamps1[0]['sections'], key=lambda x: int(x['verse_id'].split('.')[-1]))
    sections2 = sorted(timestamps2[0]['sections'], key=lambda x: int(x['verse_id'].split('.')[-1]))
    
    if len(sections1) != len(sections2):
        raise ValueError("Mismatched number of sections between the two alignments")
    
    combined = AudioSegment.empty()
    
    for sec1, sec2 in zip(sections1, sections2):
        # Extract timings and ensure they are within audio bounds
        start1, end1 = [int(float(t) * 1000) for t in sec1['timings']]
        start2, end2 = [int(float(t) * 1000) for t in sec2['timings']]
        
        end1 = min(end1, len(audio1))
        end2 = min(end2, len(audio2))
        
        # Extract and normalize segments
        seg1 = normalize(audio1[start1:end1])
        seg2 = normalize(audio2[start2:end2])
        
        combined += seg1 + seg2
    
    return combined

def main():
    parser = argparse.ArgumentParser(description="Generate interlinear audio from two language pairs.")
    parser.add_argument('--audio1', required=True, help="Path to the first audio file")
    parser.add_argument('--txt1', required=True, help="Path to the first text file")
    parser.add_argument('--audio2', required=True, help="Path to the second audio file")
    parser.add_argument('--txt2', required=True, help="Path to the second text file")
    parser.add_argument('--language1', required=True, help="Language code for the first pair")
    parser.add_argument('--language2', required=True, help="Language code for the second pair")
    parser.add_argument('--output', default='interlinear.mp3', help="Output filename (default: interlinear.mp3)")
    args = parser.parse_args()

    try:
        # Process each pair
        print("Processing first language pair...")
        timestamps1 = process_pair(args.audio1, args.txt1, args.language1)
        
        print("Processing second language pair...")
        timestamps2 = process_pair(args.audio2, args.txt2, args.language2)
        
        print("Combining audio segments...")
        final_audio = combine_audio(args.audio1, timestamps1, args.audio2, timestamps2)
        
        # Export the result
        output_path = args.output
        final_audio.export(output_path, format="mp3")
        print(f"✅ Interlinear audio saved to {output_path}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
