# Hua-Rong-Dao-Search

CSC384H1: Introduction to Artificial Intelligence\
Assignment 1

## Introduction: Hua Rong Dao

Hua Rong Dao is a sliding puzzle that is popular in China. The following page contains background information, including the story behind the puzzle and an English description of the rules.

http://chinesepuzzles.org/huarong-pass-sliding-block-puzzle/


## Input and Output File formats

Each state is represented in the following format:

- Each state is a grid of 20 characters. The grid has 5 rows with 4 characters per row.
- The empty squares are denoted by the period symbol.
- The 2x2 piece is denoted by 1.
- The single pieces are denoted by 2.
- A horizontal 1x2 piece is denoted by < on the left and > on the right. 
- A vertical 1x2 piece is denoted by ^ on the top and v on the bottom (lower cased letter v).

Here is an example of a state.

```
^^^^
vvvv
22..
11<>
1122
```

## Usage

Hrd.py contains an implementation of Depth-First Search and A* Search to solve the Hua Rong Dao puzzle. 

To run the code, use the following commands:

```
python3 hrd.py --algo astar --inputfile <input file> --outputfile <output file>    
python3 hrd.py --algo dfs --inputfile <input file> --outputfile <output file>
```

For example, to run astar on the provided basic starting state file `basic_starting_state.txt` and output the path found by astar in `astar_basic_result.txt`, use the following command:

```
python3 hrd.py --algo astar --inputfile basic_starting_state.txt --outputfile <astar_basic_result.txt>    
```
