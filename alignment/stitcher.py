import os
import sox
from typing import List
from alignment.aligner import AlignedChunk

class AudioStitcher:
    def stitch(self, chunks: List[AlignedChunk], output_path: str):
        tfm = sox.Combiner()
        
        # Build list of files in interleaved order
        input_files = []
        for chunk in chunks:
            input_files.extend([chunk.lang1_path, chunk.lang2_path])

        tfm.build(
            input_files,
            output_path,
            'concatenate'
        )
