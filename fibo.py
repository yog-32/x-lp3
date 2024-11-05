# Function to calculate Fibonacci number and count steps
def fibonacci(n, steps):
    steps[0] += 1  # Increment step count for each call
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1, steps) + fibonacci(n - 2, steps)

# Wrapper function to initialize step count and print results
def calculate_fibonacci_with_steps(n):
    steps = [0]  # List to hold step count
    result = fibonacci(n, steps)
    print(f"Fibonacci number for {n} is: {result}")
    print(f"Total steps taken: {steps[0]}")

# Test the function with an example
n = 10  # Change this value to test for different Fibonacci numbers
calculate_fibonacci_with_steps(n)
