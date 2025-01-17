from caduceus.tokenization_caduceus import CaduceusTokenizer


def test_basic_tokenization():
    tokenizer = CaduceusTokenizer(model_max_length=1024)
    
    # Test basic sequence
    sequence = "ACGT"
    tokens = tokenizer.tokenize(sequence)
    assert tokens == ["A", "C", "G", "T"]
    
    # Test token to ids
    ids = tokenizer.convert_tokens_to_ids(tokens)
    assert ids == [7, 8, 9, 10]  # Starting from 7 due to special tokens
    
    # Test ids back to tokens
    decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert decoded_tokens == ["A", "C", "G", "T"]
    
    # Test full encode/decode with special tokens
    encoded = tokenizer(sequence)
    assert encoded["input_ids"] == [*ids, tokenizer.sep_token_id]  # Includes SEP token
    
    # Test encode without special tokens
    encoded_no_special = tokenizer(sequence, add_special_tokens=False)
    assert encoded_no_special["input_ids"] == ids  # No special tokens added
    
    # Test decode with special tokens
    decoded = tokenizer.decode(encoded["input_ids"])
    assert decoded == sequence + "[SEP]"
    
    # Test decode without special tokens
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    assert decoded == sequence

def test_build_inputs_with_special_tokens():
    tokenizer = CaduceusTokenizer(model_max_length=1024)
    
    # Get some regular token IDs (A, C, G)
    regular_ids = tokenizer.convert_tokens_to_ids(["A", "C", "G"])
    
    # Test single sequence
    result = tokenizer.build_inputs_with_special_tokens(regular_ids)
    assert result == [*regular_ids, tokenizer.sep_token_id]
    
    # Test sequence pair
    ids_1 = regular_ids
    ids_2 = tokenizer.convert_tokens_to_ids(["T", "A", "C"])
    result = tokenizer.build_inputs_with_special_tokens(ids_1, ids_2)
    assert result == [*ids_1, tokenizer.sep_token_id, *ids_2, tokenizer.sep_token_id]

def test_get_special_tokens_mask():
    tokenizer = CaduceusTokenizer(model_max_length=1024)
    
    # Get some regular token IDs (A, C, G)
    regular_ids = tokenizer.convert_tokens_to_ids(["A", "C", "G"])
    
    # Test single sequence
    mask = tokenizer.get_special_tokens_mask(regular_ids, None)
    assert mask == [0, 0, 0, 1]  # 0 for regular tokens, 1 for special token (SEP)
    
    # Test sequence pair
    ids_1 = regular_ids
    ids_2 = tokenizer.convert_tokens_to_ids(["T", "A", "C"])
    mask = tokenizer.get_special_tokens_mask(ids_1, ids_2)
    assert mask == [0, 0, 0, 1, 0, 0, 0, 1]  # 0s for regular tokens, 1s for special tokens (SEPs)
    
    # Test with already_has_special_tokens=True
    special_sequence = tokenizer.build_inputs_with_special_tokens(regular_ids)
    mask = tokenizer.get_special_tokens_mask(special_sequence, None, already_has_special_tokens=True)
    assert mask == [0, 0, 0, 1]  # Only SEP token is marked as special
    
def test_unknown_characters():
    tokenizer = CaduceusTokenizer(model_max_length=1024)
    
    # Test handling of unknown characters
    sequence = "ACGTX"  # X is not in vocabulary
    tokens = tokenizer.tokenize(sequence)
    assert tokens == ["A", "C", "G", "T", "[UNK]"]
    
    # Test multiple unknown characters
    sequence = "ACGT123"
    tokens = tokenizer.tokenize(sequence)
    assert tokens == ["A", "C", "G", "T", "[UNK]", "[UNK]", "[UNK]"]

def test_case_insensitivity():
    tokenizer = CaduceusTokenizer(model_max_length=1024)
    
    # Test lowercase sequence
    sequence_lower = "acgt"
    tokens = tokenizer.tokenize(sequence_lower)
    assert tokens == ["A", "C", "G", "T"]
    
    # Test mixed case sequence
    sequence_mixed = "aCgT"
    tokens = tokenizer.tokenize(sequence_mixed)
    assert tokens == ["A", "C", "G", "T"]
    
    # Test that encoding is also case-insensitive
    encoded_upper = tokenizer("ACGT", add_special_tokens=False)
    encoded_lower = tokenizer("acgt", add_special_tokens=False)
    assert encoded_upper["input_ids"] == encoded_lower["input_ids"]