"""
Removes formatting and non-dialogue text from srt files.
"""

import argparse
from pathlib import Path

import srt

from src.utils import DATA_DIR, logger, PathList


def process_text(text):
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!',.?$ \"'%")
    # check if there are any non-alphanumeric, non-punctuation characters
    if all([c in allowed for c in text]):
        return text

    # italics means that the speaker is off-screen
    # we just strip the italic tags, but keep the text
    if "<i>" in text:
        text = text.replace("<i>", "").replace("</i>", "")

    # {y:i} is also italics
    if r"{y:i}" in text:
        text = text.replace(r"{y:i}", "")

    # handle multispeaker lines by ignoring them
    # multispeaker lines start with a dash
    # TODO: potential split these and do forced alignment
    if text.startswith("-") or text.startswith("—") or text.startswith("–"):
        return None

    # when the slash is after a punctuation, it separates two speakers
    # we ignore these lines
    toks = set([c + "/" for c in ".?!"])
    if any([x in text for x in toks]):
        return None

    if len(text) <= 0:
        return None

    # starting with a ♪ or # or * means that it's a song
    # we just ignore these lines
    if text[0] in "♪#*":
        return None

    # a "/" is also used for italics but also for music
    # conservatively, we ignore these lines
    # to detect the slash, we need to find slashes that occur after
    # whitespace or as the first character: everything else might be a date
    # or fraction
    # TODO: potentially detect whether it is italics or music
    # for now, we just ignore these lines
    if text[0] == "/" or " /" in text:
        return None

    # sound effects are in brackets or parens; we remove the text inside the brackets
    # we also remove the brackets
    # do this for all brackets in the line
    def remove_brackets(text):
        """
        iterate through the characters until we find opening bracket: ( or [
        then, continue until we find matching closing one or the end
        delete everything between the brackets, inclusive
        """
        new_text = ""
        token_stack = []
        for c in text:
            if c in "([":
                token_stack.append(c)
                continue
            if token_stack:
                if c == ")" and token_stack[-1] == "(":
                    token_stack.pop()
                    continue
                if c == "]" and token_stack[-1] == "[":
                    token_stack.pop()
                    continue
                continue
            new_text += c
        return new_text.strip()
        
    if "(" in text or "[" in text:
        text = remove_brackets(text)

    # if there is a colon after all caps, it's a speaker label. We remove it
    # TODO: sometimes, the speaker label is not in all cap. There is no way
    # to distinguish between speaker label and dialogue in this case
    if ":" in text:
        label, dialogue = text.split(":", 1)
        if label.isupper():
            text = dialogue

    text = text.strip()
    if not text:
        return None
    return text


def process_srt_file(srt_file, output_dir):
    srt_generator = srt.parse(open(srt_file).read())
    logger.info(f"Processing {srt_file}")
    entries = []
    for entry in srt_generator:
        text = " ".join(entry.content.strip().split("\n"))
        text = process_text(text)
        if text is None:
            continue
        new_entry = srt.Subtitle(
            index=entry.index,
            start=entry.start,
            end=entry.end,
            content=text
        )
        entries.append(new_entry)
    return entries


def main(args):
    srt_files = PathList(args.srt_files)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for srt_file in srt_files:
        entries = process_srt_file(srt_file, output_dir)
        # write the cleaned srt file to the output directory
        with open(output_dir / srt_file.name, "w") as f:
            f.write(srt.compose(entries))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srt_files", nargs="+", help="The srt file to clean")
    parser.add_argument("--output_dir", type=Path, help="The directory to save the cleaned srt file to", default=DATA_DIR / "cleaned_srt")
    args = parser.parse_args()
    main(args)