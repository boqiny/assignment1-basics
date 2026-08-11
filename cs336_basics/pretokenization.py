import os
from typing import BinaryIO
import multiprocessing
import regex as re
from collections import defaultdict, Counter
import functools

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(bound: tuple[int, int], input_path: str, special_tokens: list[str]) -> dict[str, int]:
        # desired output example: {low: 5, lower: 2, widest: 3, newest: 6}
        start, end = bound[0], bound[1]
        counts = defaultdict(int)
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if not special_tokens:
                splits = [chunk]
            else:
                splits = re.split('|'.join(re.escape(token) for token in special_tokens), chunk)
            for split in splits:
                matches = re.finditer(PAT, split)
                for match in matches:
                    counts[match.group()] += 1
        return counts

# if __name__ == "__main__":
#     with open(FILE_LOCATION, "rb") as f:
#         num_processes = 4
#         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#         p = multiprocessing.Pool(processes=num_processes)
#         boundaries_list = list(zip(boundaries[:-1], boundaries[1:]))
#         result = p.map(pre_tokenize, boundaries_list)
#         pretokenization_counter = Counter()
#         for r in result:
#             pretokenization_counter.update(r)
#         print(str(pretokenization_counter))
        
    
def count_pretokenization(input_path: str, num_processes: int, special_tokens: list[str]) -> dict[str, int]:

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        with multiprocessing.Pool(processes=num_processes) as p:
            boundaries_list = list(zip(boundaries[:-1], boundaries[1:]))
            result = p.map(
                functools.partial(pre_tokenize, input_path=input_path, special_tokens=special_tokens),
                boundaries_list
                )
            pretokenization_counter = Counter()
            for r in result:
                pretokenization_counter.update(r)
        return pretokenization_counter