from typing import Iterable
import string

def is_palindrom(
    sequence,
    case_sensitive: bool,
    type_sensitive: bool,
    sentence_sensitive: bool
    ):

    if isinstance(sequence, Iterable):
        if isinstance(sequence, str):
            seq = sequence
            seq = seq.translate(str.maketrans('', '', string.punctuation))
            type_sensitive = False

            if case_sensitive is False:
                seq = seq.lower()

            if sentence_sensitive:
                seq = seq.split(" ")
            else:
                seq = seq.translate(str.maketrans('', '', " "))

        else:
            if type_sensitive:
                types = [type(item) for item in sequence]
            seq = list(map(str,sequence))
            if case_sensitive is False:
                seq = [item.lower() for item in seq]
    else:
        seq = str(sequence)

    if len(seq) < 2:
        raise ValueError

    left_idx = 0
    right_idx = -1
    n_checks = round(len(seq)/2)
    
    for _ in range(n_checks):
        if seq[left_idx] != seq[right_idx]:
            return False
        if type_sensitive:
            if types[left_idx] != types[right_idx]:
                return False
        else:
            left_idx, right_idx = left_idx+1, right_idx-1
            continue      
    return True

test_cases = {
    "check empty string":{
        "case": '',
        "case_sensitive": True,
        "type_sensitive": False,
        "sentence_sensitive": False,
        "expectation": ValueError()
        },

    "check single integer":{
        "case": 1,
        "case_sensitive": True,
        "type_sensitive": False,
        "sentence_sensitive": False,
        "expectation": ValueError()
        },

    "check palindromed float":{
    "case": 1.1,
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check not palindromed float":{
    "case": 1.21,
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check palindromed intiger":{
    "case": 1111,
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check not palindromed intiger":{
    "case": 123456,
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check palindromed lower string, case sensitive":{
    "case": "abba",
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check palindromed lower string, case insensitive":{
    "case": "abba",
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check palindromed title string, case sensitive":{
    "case": "Abba",
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check palindromed title string, case insensitive":{
    "case": "Abba",
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check not palindromed lower string, case sensitive":{
    "case": "apple",
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check not palindromed lower string, case insensitive":{
    "case": "apple",
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check not palindromed list of intigers":{
    "case": [1,2,3],
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check palindromed list of intigers":{
    "case": [1,2,3,2,1],
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check palindromed list of strings":{
    "case": ["kot","pies","pies","kot"],
    "case_sensitive": False,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check not palindromed list of strings with capital and lower letters":{
    "case": ["Kot","pies","pies","kot"],
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check list with not palindromed types":{
    "case": [1,2,3,"2","1"],
    "case_sensitive": True,
    "type_sensitive": False,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check list with not palindromed types, type sensitive":{
    "case": [1,2,3,"2","1"],
    "case_sensitive": True,
    "type_sensitive": True,
    "sentence_sensitive": False,
    "expectation": False
    },

    "check palindrome string, by characters without punctuations":{
    "case": "Mr. Owl ate my metal worm",
    "case_sensitive": False,
    "type_sensitive": True,
    "sentence_sensitive": False,
    "expectation": True
    },

    "check palindrome string, by words":{
    "case": "Ho, Ho, Ho, Ho",
    "case_sensitive": False,
    "type_sensitive": True,
    "sentence_sensitive": True,
    "expectation": True
    },

}

for note, test in test_cases.items():
    if isinstance(test['expectation'], Exception):
        try:
            result = is_palindrom(
                sequence = test['case'],
                case_sensitive=test['case_sensitive'],
                type_sensitive=test['type_sensitive'],
                sentence_sensitive=test['sentence_sensitive']
                )
            print(
                f"Expected {test['expectation']!r} for case {test['case']} got {result!r}"
                )
        except ValueError:
            continue
    else:
        result = is_palindrom(
                sequence = test['case'],
                case_sensitive=test['case_sensitive'],
                type_sensitive=test['type_sensitive'],
                sentence_sensitive=test['sentence_sensitive']
                ) 
        if result != test['expectation']:
            print(f"test: {note}")
            print(
                f"Expected {test['expectation']!r} for case {test['case']} got {result!r}"
                )