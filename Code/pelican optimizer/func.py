import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to Logger
sys.stdout = Logger("F30_results.txt")

# Example usage
print("Function: F30")
print("Best Score: 3200.0")
print("Worst Score: 3200.0")
print("Mean Score: 3200.0")
print("Standard Deviation: 0")
print("All Best Scores:")
for i in range(50):
    print("3200.0", end = ", ")
