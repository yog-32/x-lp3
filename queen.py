# Define the size of the board (8x8 for the 8-Queens problem)
N = 8

# Initialize the board with all zeros (no queens placed)
board = [[0] * N for _ in range(N)]

# Place the first Queen in the first column
board[0][0] = 1

# Function to check if a Queen can be placed on the board at a specific row and column
def is_safe(board, row, col):
    # Check this row on the left side
    for i in range(col):
        if board[row][i] == 1:
            return False
    
    # Check the upper diagonal on the left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    # Check the lower diagonal on the left side
    for i, j in zip(range(row, N), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    return True

# Function to solve the 8-Queens problem using backtracking
def solve_queens(board, col):
    # If all queens are placed, return True
    if col >= N:
        return True
    
    # Try placing a queen in all rows one by one
    for i in range(N):
        if is_safe(board, i, col):
            # Place the queen
            board[i][col] = 1
            
            # Recur to place the rest of the queens
            if solve_queens(board, col + 1):
                return True
            
            # If placing queen in the above row doesn't lead to a solution
            # then remove the queen (backtrack)
            board[i][col] = 0
    
    # If no queen can be placed in any row in this column, return False
    return False

# Start solving from the second column since the first Queen is already placed
solve_queens(board, 1)

# Display the final board
for row in board:
    print(row)
