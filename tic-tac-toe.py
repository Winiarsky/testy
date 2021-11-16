import re

def tic_tac_toe_winner(baord: str) -> str:
    if len(baord) != 9:
        print("board lenght must equal 9")
        raise ValueError
    for char in set(baord):
        if char not in "XO ":
            print(f"invalid character- {char}")
            raise ValueError

    check_areas = [
        baord[0:3],  # row 1
        baord[3:6],  # row 2
        baord[6:10],  # row 3
        baord[0::3],  # col 1
        baord[1::3],  # col 2
        baord[2::3],  # col 3
        baord[0::3],  # col 3
        baord[0::4],  # diag 1
        baord[2:7:2],  # diag 2
    ]

    for area in check_areas:
        if re.match("XXX|OOO",area):
            winner = area[0]
            print(f"Winner is {winner}!")
            return winner
    print("Draw!")
    return None



test_cases = {
    'XO  X O X': 'X',
    'OX  O X O': 'O',
    'XXOOXXXOO': None,
    'XX': ValueError,
    'lX': ValueError,

}
for board, expectation in test_cases.items():
    if isinstance(expectation, Exception):
        try:
            response = tic_tac_toe_winner(board)
            print(f'Expected {expectation!r} for {board!r} got {response!r}')
        except expectation:
            pass




