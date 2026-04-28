import json
import time
from pathlib import Path
import resource

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

def _bytes_to_gpt2_string(token: bytes) -> str:
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token)


def _save_bpe_result(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = {
        _bytes_to_gpt2_string(token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    merges_path = output_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            left_s = _bytes_to_gpt2_string(left)
            right_s = _bytes_to_gpt2_string(right)
            f.write(f"{left_s} {right_s}\n")


def _print_bpe_summary(
    name: str,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    elapsed_time: float,
) -> None:
    longest_token = max(vocab.values(), key=len)
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print()
    print("=" * 80)
    print(f"BPE experiment: {name}")
    print("=" * 80)
    print(f"vocab size: {len(vocab)}")
    print(f"number of merges: {len(merges)}")
    print(f"training time: {elapsed_time:.2f} seconds")
    print(f"max RSS: {max_rss_mb:.2f} MB")
    print(f"longest token length: {len(longest_token)} bytes")
    print(f"longest token repr: {repr(longest_token)}")
    print(
        "longest token decoded:",
        longest_token.decode("utf-8", errors="replace"),
    )
    print("=" * 80)
    print()


def test_train_bpe_tinystories():
    input_path = Path(
        "/root/WorkSpace/cs336/assignment1-basics/data/"
        "TinyStoriesV2-GPT4-train.txt"
    )
    output_dir = Path(
        "/root/WorkSpace/cs336/assignment1-basics/bpe_outputs/"
        "tinystories_10k"
    )

    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    _save_bpe_result(vocab, merges, output_dir)
    _print_bpe_summary(
        name="tinystories_10k",
        vocab=vocab,
        merges=merges,
        elapsed_time=end_time - start_time,
    )

    assert len(vocab) == 10000
    assert b"<|endoftext|>" in set(vocab.values())
    assert len(merges) == 10000 - 256 - 1
    assert (output_dir / "vocab.json").exists()
    assert (output_dir / "merges.txt").exists()


def test_train_bpe_expts_owt():
    input_path = Path(
        "/root/WorkSpace/cs336/assignment1-basics/data/"
        "owt_train.txt"
    )
    output_dir = Path(
        "/root/WorkSpace/cs336/assignment1-basics/bpe_outputs/"
        "owt_32k"
    )

    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    _save_bpe_result(vocab, merges, output_dir)
    _print_bpe_summary(
        name="owt_32k",
        vocab=vocab,
        merges=merges,
        elapsed_time=end_time - start_time,
    )

    assert len(vocab) == 32000
    assert b"<|endoftext|>" in set(vocab.values())
    assert len(merges) == 32000 - 256 - 1
    assert (output_dir / "vocab.json").exists()
    assert (output_dir / "merges.txt").exists()