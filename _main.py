import subprocess
import os
import sys

# Ensure console uses UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

def run_script(script_path):
    try:
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        print("\nOutput:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running script: {e}")


def main():
    scripts = {
        '1': os.path.join('Clustering', 'DBSCAN.py'),
        '2': os.path.join('Clustering', 'K_Means.py'),
        '3': os.path.join('Regression', 'Regression.py'),
        '4': os.path.join('NeuralNetwork', 'NeuralNetwork.py'),
        '5': os.path.join('RandomForest', 'RandomClassifier.py'),
    }

    while True:
        print("\nChoose a task:")
        print("1 - Clustering - DBSCAN")
        print("2 - Clustering - K-Means")
        print("3 - Regression")
        print("4 - MLP Multilabel")
        print("5 - Random Forest Multilabel")
        print("q - Quit")

        choice = input("Enter your choice (1-5 or q to quit): ").strip().lower()

        if choice == 'q':
            print("Exiting program.")
            break
        elif choice in scripts:
            run_script(scripts[choice])
        else:
            print("Invalid choice. Please enter a number from 1 to 5, or 'q' to quit.")

if __name__ == "__main__":
    main()
