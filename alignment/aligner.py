import os
import torch
from typing import Dict, List
from dataclasses import dataclass
from mms.align_utils import get_alignments, get_model_and_dict

@dataclass
class AlignedChunk:
    lang1_path: str
    lang2_path: str
    verse_number: int

class AudioAligner:
    def __init__(self):
        self.model, self.dictionary = get_model_and_dict()
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def process(self, audio1_path: str, text1_path: str, 
                audio2_path: str, text2_path: str,
                lang1: str, lang2: str) -> List[AlignedChunk]:
        # Process first language
        lang1_chunks = self._align_single(audio1_path, text1_path, lang1)
        lang2_chunks = self._align_single(audio2_path, text2_path, lang2)

        if len(lang1_chunks) != len(lang2_chunks):
            raise ValueError("Mismatched verse counts between languages")

        # Pair chunks
        aligned_chunks = []
        for i, (chunk1, chunk2) in enumerate(zip(lang1_chunks, lang2_chunks)):
            aligned_chunks.append(AlignedChunk(
                lang1_path=chunk1,
                lang2_path=chunk2,
                verse_number=i + 1
            ))

        return aligned_chunks

    def _align_single(self, audio_path: str, text_path: str, lang: str) -> List[str]:
        # Implementation using your existing alignment code
        # Returns list of paths to individual verse chunks
        pass
