# audio-interlinearify

A tool that uses Meta's MMS model to create interlinear audio bibles, in potentially 1,000+ different languages.

The idea is explained in depth in [this blog post](https://joshmuller.ca/writings/2024/interlinear-audiobible)

# Installation

## 1. Clone this repo 

```sh
git clone git@github.com:theJoshMuller/audio-interlinearify.git
```

## 2. Install Python Requirements

I've found working in a virtual enviromentment to be really helpful:

```sh
python3.11 -m venv venv
source venv/bin/activate
```

Once your environment is in place, install the requirements:

```sh
pip install -r requirements.txt
```

You'll also likely need to make sure that, whatever system you're using has `ffmpeg` and `sox` installed.

On Arch Linux that's:

```sh
sudo pacman -S ffmpeg sox
```

Your installation command will depend on your system.


## 3. Usage

To make an interlinear audio bible, you'll need a text (`txt`) and audio (`mp3`) file for each language, for each chapter you want to create an interlinear audio file for.


The text files will need to have each "segment" (verse, sentence, etc.) that you want separated by lines, and _each txt file will need to have the same number of lines._

Example files are provided in `./sample_data`

Here's the syntax for the command:

```sh
python interlinearify.py \
    --audio1 "./sample_data/ISA_061.eng.mp3" \
    --txt1 "./sample_data/ISA_061.eng.txt" \
    --audio2 "./sample_data/ISA_061.heb.mp3" \
    --txt2 "./sample_data/ISA_061.heb.txt" \
    --language1 "eng" \
    --language2 "heb" \
    --output "ISA_061.eng-heb-interlinear.mp3"
```

Language options can be found in `./data/mms_languages.json`

# 4. Contributions Welcome

This is just a fun side project. If you want to contribute, feel free!

Thanks to [Trent Cowden](https://trentcowden.com/) for building out [TimeStampAudio CLI](https://github.com/kingdomstrategies/waha-ai-timestamper-cli), and thanks to Meta for releasing their [MMS](https://ai.meta.com/blog/multilingual-model-speech-recognition/) model.
