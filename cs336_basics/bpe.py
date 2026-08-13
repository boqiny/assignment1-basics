from cs336_basics.pretokenization import count_pretokenization
from collections import defaultdict

def count_adjacent_pairs(pretokenization_bytes_dict: defaultdict(int)) -> dict[tuple[bytes, bytes], int]:
    # Input example: (b' ', b't', b'w', b'i', b'r', b'l', b'e', b'r'): 2counts = defaultdict(int)
    counts = defaultdict(int)
    for bytes_tuple, count in pretokenization_bytes_dict.items():
        for byte1, byte2 in zip(bytes_tuple, bytes_tuple[1:]):
            counts[(byte1, byte2)] += count
    return counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - len(special_tokens) - 256
    # print("num_merges: ", num_merges)
    pretokenization_counter = count_pretokenization(input_path, 4, special_tokens)
    pretokenization_bytes_dict = defaultdict(int)
    for k,v in pretokenization_counter.items():
        indices_list = list(k.encode("utf-8"))
        pretokenization_bytes_dict[tuple([bytes([i]) for i in indices_list])] += v
    # print("pretokenization_bytes_dict: ", pretokenization_bytes_dict)
    # pair_counts = count_adjacent_pairs(pretokenization_bytes_dict)
    # print(pair_counts)

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    for i in range(len(special_tokens)):
        vocab[vocab_size - len(special_tokens) + i] = special_tokens[i].encode('utf-8')

    for i in range(num_merges):
        # Find the most common pair
        pair_counts = count_adjacent_pairs(pretokenization_bytes_dict)
        most_common_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
        new_bytes = most_common_pair[0] + most_common_pair[1]
        # Add new token to vocab
        vocab[256 + i] = new_bytes
        # Add pair to merges
        merges.append(most_common_pair)
        pretokenization_bytes_dict = merge(pretokenization_bytes_dict, most_common_pair[0], most_common_pair[1])
        # print("pretokenization_bytes_dict: ", pretokenization_bytes_dict)
    
    print("len vocab:", len(vocab))
    return (vocab, merges)

def merge(pretokenization_bytes_dict: defaultdict(int), byte1, byte2):
    new_pretokenization_bytes_dict = defaultdict(int)
    for k,v in pretokenization_bytes_dict.items():
        new_key = []
        i = 0
        while i < len(k):
            if i + 1 < len(k) and k[i] == byte1 and k[i+1] == byte2:
                new_key.append(byte1+byte2)
                i += 2
            else:
                new_key.append(k[i])
                i += 1
        new_pretokenization_bytes_dict[tuple(new_key)] = v
    return new_pretokenization_bytes_dict
        

if __name__ =="__main__":
    train_bpe("../data/bpe_toy.txt", 269, special_tokens = ['<|endoftext|>'])