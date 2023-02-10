from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
import itertools

# =============================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str or None
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single,
                                       self.coord_x, self.coord_y,
                                       self.orientation)

    def piece_type(self):
        """
        :return: the type of piece
        """

        if self.is_goal:
            return 'this piece is the 2x2 goal piece'
        if self.is_single:
            return 'this piece is a 1x1 piece'
        if self.orientation is not None:
            return 'this is a 2x1 piece with orientation ' + self.orientation \
                   + '. The upper left corner is at (' + str(self.coord_x) + \
                   ", " + str(self.coord_y) + ')'

    def move_piece(self, x_direction: int, y_direction: int):
        """
        Moves the piece in x_direction and y_direction.

        Assumption: the move is valid / on the board.
        NOTE: Only moves one piece. Other pieces must be handled
        manually.
        :return:
        """
        self.coord_x += x_direction
        self.coord_y += y_direction


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location
        information.
        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self, file):
        """
        Print out the current board.
        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='', file=file, flush=True)
            print(file=file, flush=True)
        print(file=file, flush=True)

    def find_empty(self) -> [(int, int), (int, int)]:
        """
        Returns all empty squares on the current board
        :return: list of the co-ordinates of the empty squares
        >>> board1 = read_from_file("test1_easy.txt")
        >>> board1.find_empty()
        [(2, 3), (2, 4)]
        """
        empty_spaces = []

        for x in range(self.width):
            for y in range(self.height):
                if self.grid[y][x] == '.':
                    empty_spaces.append((x, y))
        return empty_spaces

    def get_piece_at_cords(self, x, y) -> Piece:
        """
        Finds the piece with co-ordinates (x, y)
        :param x: The required x co-ordinate
        :param y: The required y co-ordinate
        :return: The piece at (x, y)
        """
        for piece in self.pieces:
            if piece.coord_x == x and piece.coord_y == y:
                return piece


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.


def is_solved(board: Board) -> bool:
    """
    Returns true if board is solved, and false otherwise.

    :param board: the board in question
    :return: True if solved, False otherwise
    >>> board1 = read_from_file("CSC384/A1/test1_easy.txt")
    >>> is_solved(board1)
    False
    >>> board2 = read_from_file("CSC384/A1/test1_solved.txt")
    >>> is_solved(board2)
    True
    """
    for piece in board.pieces:
        if piece.is_goal:
            return (piece.coord_x, piece.coord_y) == (1, 3)


def get_heuristic_values(board: Board) -> int:
    """
    Returns the heuristic value of the current board state
    :param board: the board
    :return: the heuristic value
    >>> board1 = read_from_file("CSC384/A1/basic_starting_state.txt")
    >>> get_heuristic_values(board1)
    3
    >>> board2 = read_from_file("CSC384/A1/test1_easy.txt")
    >>> get_heuristic_values(board2)
    1
    >>> board3 = read_from_file("CSC384/A1/test1_solved.txt")
    >>> get_heuristic_values(board3)
    0
    """
    if is_solved(board):
        return 0
    for piece in board.pieces:
        if piece.is_goal:
            return abs((piece.coord_x - 1) + (piece.coord_y - 3))


def single_proximity_to_empty(piece: Piece, empty_squares: [(int, int),
                                                            (int, int)]) -> int:
    """
    Helper function for generate_successors.

    Finds the number of directions that a single 1x1 piece can be moved.
    :param piece: a 1x1 single piece
    :param empty_squares: the empty squares on the current board
    :return: the number of directions piece can be moved
    """
    y = piece.coord_y
    x = piece.coord_x

    total = 0

    # move right
    if (x + 1, y) in empty_squares:
        total += 1
    # move left
    if (x - 1, y) in empty_squares:
        total += 1
    # move up
    if (x, y - 1) in empty_squares:
        total += 1
    # move down
    if (x, y + 1) in empty_squares:
        total += 1
    return total


def goal_proximity_to_empty(piece: Piece, empty_squares: [(int, int),
                                                          (int, int)]) -> int:
    """
    Helper function for generate_successors.

    Finds the number of directions that the goal piece can be moved.
    :param piece: the goal piece
    :param empty_squares: the empty squares on the current board
    :return: the number of directions piece can be moved
    """

    x = piece.coord_x
    y = piece.coord_y

    total = 0

    # Move right
    if (x + 2, y) in empty_squares and (x + 2, y + 1) in empty_squares:
        total += 1
    # Move left
    if (x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares:
        total += 1
    # Move up
    if (x, y - 1) in empty_squares and (x + 1, y - 1) in empty_squares:
        total += 1
    # Move down
    if (x, y + 2) in empty_squares and (x + 1, y + 2) in empty_squares:
        total += 1
    return total


def horizontal_moves(piece: Piece, empty_squares: [(int, int),
                                                   (int, int)]) -> int:
    """
    Helper function for generate_successors.

    Finds the number of directions that a horizontal 2x1 piece can be moved.
    :param piece: a horizontal 2x1 piece
    :param empty_squares: the empty squares on the current board
    :return: the number of directions piece can be moved
    """
    y = piece.coord_y
    x = piece.coord_x

    total = 0

    # move right
    if (x + 2, y) in empty_squares:
        total += 1
    # move left
    if (x - 1, y) in empty_squares:
        total += 1
    # move up
    if (x, y - 1) in empty_squares and (x + 1, y - 1) in empty_squares:
        total += 1
    # move down
    if (x, y + 1) in empty_squares and (x + 1, y + 1) in empty_squares:
        total += 1

    return total


def vertical_moves(piece: Piece, empty_squares: [(int, int),
                                                 (int, int)]) -> int:
    """
    Helper function for generate_successors.

    Finds the number of directions that a vertical 2x1 piece can be moved.
    :param piece: a vertical 2x1 piece
    :param empty_squares: the empty squares on the current board
    :return: the number of directions piece can be moved
    """
    y = piece.coord_y
    x = piece.coord_x

    total = 0
    # move right
    if (x + 1, y) in empty_squares and (x + 1, y + 1) in empty_squares:
        total += 1
    # move left
    if (x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares:
        total += 1
    # move up
    if (x, y - 1) in empty_squares:
        total += 1
    # move down
    if (x, y + 2) in empty_squares:
        total += 1
    return total


def single_direction_to_move(piece: Piece,
                             empty_squares: [(int, int), (int, int)]) \
        -> list[[int, int]]:
    """
    Helper function for generate_successors.

    Finds all viable directions that a single 1x1 piece can be moved.
    :param piece: a single 1x1 piece
    :param empty_squares: the empty squares on the current board
    :return: a list of the directions to move piece
    """
    y = piece.coord_y
    x = piece.coord_x
    directions = []

    # move right
    if (x + 1, y) in empty_squares:
        directions.append([1, 0])
    # move left
    if (x - 1, y) in empty_squares:
        directions.append([-1, 0])
    # move up
    if (x, y - 1) in empty_squares:
        directions.append([0, -1])
    # move down
    if (x, y + 1) in empty_squares:
        directions.append([0, 1])

    return directions


def goal_direction_to_move(piece: Piece,
                           empty_squares: [(int, int), (int, int)]) \
        -> list[[int, int]]:
    """
    Helper function for generate_successors.

    Finds all viable directions that the goal piece can be moved.
    :param piece: the goal piece
    :param empty_squares: the empty squares on the current board
    :return: a list of the directions to move piece
    """
    y = piece.coord_y
    x = piece.coord_x
    directions = []

    # Move right
    if (x + 2, y) in empty_squares and (x + 2, y + 1) in empty_squares:
        directions.append([1, 0])
    # Move left
    if (x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares:
        directions.append([-1, 0])
    # Move up
    if (x, y - 1) in empty_squares and (x + 1, y - 1) in empty_squares:
        directions.append([0, -1])
    # Move down
    if (x, y + 2) in empty_squares and (x + 1, y + 2) in empty_squares:
        directions.append([0, 1])

    return directions


def horizontal_direction_to_move(piece: Piece,
                                 empty_squares: [(int, int), (int, int)]) \
        -> list[[int, int]]:
    """
    Helper function for generate_successors.

    Finds all viable directions that a horizontal 2x1 piece can be moved.
    :param piece: a horizontal 2x1 piece
    :param empty_squares: the empty squares on the current board
    :return: a list of the directions to move piece
    """

    y = piece.coord_y
    x = piece.coord_x
    directions = []

    # move right
    if (x + 2, y) in empty_squares:
        directions.append([1, 0])
    # move left
    if (x - 1, y) in empty_squares:
        directions.append([-1, 0])
    # move up
    if (x, y - 1) in empty_squares and (x + 1, y - 1) in empty_squares:
        directions.append([0, -1])
    # move down
    if (x, y + 1) in empty_squares and (x + 1, y + 1) in empty_squares:
        directions.append([0, 1])

    return directions


def vertical_direction_to_move(piece: Piece,
                               empty_squares: [(int, int), (int, int)]) \
        -> list[[int, int]]:
    """
    Helper function for generate_successors.

    Finds all viable directions that a vertical 2x1 piece can be moved.
    :param piece: a vertical 2x1 piece
    :param empty_squares: the empty squares on the current board
    :return: a list of the directions to move piece
    """
    y = piece.coord_y
    x = piece.coord_x
    directions = []

    # move right
    if (x + 1, y) in empty_squares and (x + 1, y + 1) in empty_squares:
        directions.append([1, 0])
    # move left
    if (x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares:
        directions.append([-1, 0])
    # move up
    if (x, y - 1) in empty_squares:
        directions.append([0, -1])
    # move down
    if (x, y + 2) in empty_squares:
        directions.append([0, 1])
    return directions


def create_state(original_state: State, piece: Piece, directions):
    """
    Helper function for generate_successors.

    Creates a new state for each viable move of piece

    :param original_state: the original state
    :param piece: the piece being moved
    :param directions: the directions to move the piece
    :return: a list of new states where piece has been moved
    """
    result = []

    for direction in directions:
        copy = deepcopy(original_state.board)
        piece_to_move = copy.get_piece_at_cords(piece.coord_x,
                                                piece.coord_y)
        piece_to_move.move_piece(direction[0], direction[1])
        new_board = Board(copy.pieces)
        new_state = State(new_board, get_heuristic_values(new_board),
                          original_state.depth + 1, original_state)
        result.append(new_state)
    return result


def generate_successors(state: State) -> list[State]:
    """
    Generates all possible moves from current board state.

    :param state: the original state
    :return: a list of new states where all movable pieces have been moved
    >>> board1 = read_from_file("basic_starting_state.txt")
    >>> state1 = State(board1, get_heuristic_values(board1), 0, None)
    >>> board2 = read_from_file("test1_easy.txt")
    >>> state2 = State(board2, get_heuristic_values(board2), 0, None)
    >>> result = generate_successors(state2)
    >>> result[0].board.display()
    2^22
    2v<>
    <><>
    .11^
    .11v
    >>> is_solved(result[0].board)
    True
    >>> result[1].board.display()
    2^22
    2v<>
    <><>
    11^.
    11v.
    >>> board3 = read_from_file("mid_state.txt")
    >>> state3 = State(board3, get_heuristic_values(board3), 0, None)
    >>> result = generate_successors(state3)
    >>> result[0].board.display()
    ^11^
    v11v
    ^..^
    v<>v
    2222
    >>> result[1].board.display()
    ^11^
    v11v
    ^<>^
    v2.v
    2.22
    >>> result[2].board.display()
    ^11^
    v11v
    ^<>^
    v.2v
    22.2
    >>> board4 = read_from_file("goal_piece_test.txt")
    >>> state4 = State(board4, get_heuristic_values(board4), 0, None)
    >>> result2 = generate_successors(state4)
    >>> result2[1].board.display()
    <><>
    ^11^
    v11v
    ^<>^
    v..v
    >>> board5 = read_from_file("vertical_piece_test.txt")
    >>> state5 = State(board5, get_heuristic_values(board5), 0, None)
    >>> result = generate_successors(state5)
    >>> board6 = read_from_file("horizontal_piece_test.txt")
    >>> state6 = State(board6, get_heuristic_values(board6), 0, None)
    """
    empty_squares = state.board.find_empty()
    states = []

    # iterate through all pieces in current state
    for piece in state.board.pieces:

        # if the piece is a single 1x1 piece
        if piece.is_single and single_proximity_to_empty(piece,
                                                         empty_squares):
            directions = single_direction_to_move(piece, empty_squares)
            states.extend(create_state(state, piece, directions))
        # if the piece is the goal piece
        elif piece.is_goal and goal_proximity_to_empty(piece, empty_squares):
            directions = goal_direction_to_move(piece, empty_squares)
            states.extend(create_state(state, piece, directions))
        # if the piece is the 2x1 piece in horizontal orientation
        elif piece.orientation == 'h' and \
                horizontal_moves(piece, empty_squares):
            directions = horizontal_direction_to_move(piece, empty_squares)
            states.extend(create_state(state, piece, directions))
        # if the piece is the 2x1 piece in vertical orientation
        elif piece.orientation == 'v' and \
                vertical_moves(piece, empty_squares):
            directions = vertical_direction_to_move(piece, empty_squares)
            states.extend(create_state(state, piece, directions))

    return states


def get_solution(goal_state: State) -> list[State]:
    """
    Returns a list of states from the initial state to the goal_state
    :param goal_state:
    :return:
    >>> board1 = read_from_file("CSC384/A1/test1_easy.txt")
    >>> state1 = State(board1, get_heuristic_values(board1), 0, None)
    >>> board2 = read_from_file("CSC384/A1/test1_solved.txt")
    >>> state2 = State(board2, get_heuristic_values(board2), 1, state1)
    >>> result = get_solution(state2)
    >>> result[0].board.display()
    2^22
    2v<>
    <><>
    11.^
    11.v
    >>> result[1].board.display()
    2^22
    2v<>
    <><>
    .11^
    .11v
    """
    result = [goal_state]
    curr_state = goal_state

    while curr_state.parent is not None:
        result.insert(0, curr_state.parent)
        curr_state = curr_state.parent
    return result


def dfs_algorithm(initial_state: State) -> State:
    """
    DFS algorithm
    :param initial_state:
    :return:
    >>> board1 = read_from_file("test1_easy.txt")
    >>> state1 = State(board1, get_heuristic_values(board1), 0, None)
    >>> dfs_algorithm(state1).board.display()
    2^22
    2v<>
    <><>
    .11^
    .11v
    >>> dfs_algorithm(state1).depth
    1
    >>> board2 = read_from_file("basic_starting_state.txt")
    >>> state2 = State(board2, get_heuristic_values(board2), 0, None)
    >>> result = dfs_algorithm(state2)
    >>> result.board.display()
    >>> result.depth
    """
    frontier = [initial_state]
    seen = set()

    while frontier:
        curr_state = frontier.pop()
        seen.add(hash(str(curr_state.board.grid)))

        successors = generate_successors(curr_state)
        for successor in successors:
            if is_solved(successor.board):
                return successor
            else:
                if hash(str(successor.board.grid)) not in seen:
                    frontier.append(successor)


def astar_algorithm(initial_state) -> State:
    """
    astar algorithm
    :param initial_state:
    :return:
    >>> board1 = read_from_file("test1_easy.txt")
    >>> state1 = State(board1, get_heuristic_values(board1), 0, None)
    >>> astar_algorithm(state1).board.grid == read_from_file("test1_solved.txt").grid
    True
    >>> astar_algorithm(state1).depth
    1
    >>> board2 = read_from_file("basic_starting_state.txt")
    >>> state2 = State(board2, get_heuristic_values(board2), 0, None)
    >>> result = astar_algorithm(state2)
    """
    seen = set()
    seq = itertools.count().__next__

    total = initial_state.depth + get_heuristic_values(initial_state.board)
    frontier = [(total, seq(), initial_state)]

    while frontier:
        state = heappop(frontier)[2]
        if is_solved(state.board):
            return state
        if hash(str(state.board.grid)) not in seen:
            seen.add(hash(str(state.board.grid)))

            successors = generate_successors(state)

            for successor in successors:
                total = successor.depth + get_heuristic_values(
                    successor.board)
                heappush(frontier, (total, seq(), successor))


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if not g_found:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)

    if args.algo == 'astar':

        original_state = astar_algorithm(
            State(board, get_heuristic_values(board), 0, None))

        file = open(args.outputfile, 'w')

        for i in range(len(get_solution(original_state))):
            print(get_solution(original_state)[i].board.display(file),
                  flush=True)
        file.flush()
        file.close()
    elif args.algo == 'dfs':
        original_state = dfs_algorithm(
            State(board, get_heuristic_values(board), 0, None))

        file = open(args.outputfile, 'w')

        for i in range(len(get_solution(original_state))):
            print(get_solution(original_state)[i].board.display(file),
                  flush=True)

        file.flush()
        file.close()
