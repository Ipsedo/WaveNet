import argparse
import soundfile as sf

from tqdm import tqdm

from os.path import basename, join

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert audio")

    parser.add_argument("input_files", type=str, nargs="+")
    parser.add_argument("output_dir", type=str)

    parser.add_argument("-r", type=int, default=16000)

    args = parser.parse_args()

    files = args.input_files

    for f in tqdm(files):
        raw_audio, sample_rate = sf.read(f)

        sf.write(
            join(args.output_dir, basename(f)),
            raw_audio, samplerate=args.r, subtype='PCM_16'
        )
