import subprocess
import sys
import argparse

python_path = sys.executable

file_names = [
    'Abdulkader/vrpnc1a.txt', 'Abdulkader/vrpnc1b.txt',
    'Abdulkader/vrpnc2a.txt', 'Abdulkader/vrpnc2b.txt',
    'Abdulkader/vrpnc3a.txt', 'Abdulkader/vrpnc3b.txt',
    'Abdulkader/vrpnc4a.txt', 'Abdulkader/vrpnc4b.txt',
    'Abdulkader/vrpnc5a.txt', 'Abdulkader/vrpnc5b.txt',
    'Abdulkader/vrpnc6a.txt', 'Abdulkader/vrpnc6b.txt',
    'Abdulkader/vrpnc7a.txt', 'Abdulkader/vrpnc7b.txt',
    'Abdulkader/vrpnc8a.txt', 'Abdulkader/vrpnc8b.txt',
    'Abdulkader/vrpnc9a.txt', 'Abdulkader/vrpnc9b.txt',
    'Abdulkader/vrpnc10a.txt', 'Abdulkader/vrpnc10b.txt',
    'Abdulkader/vrpnc11a.txt', 'Abdulkader/vrpnc11b.txt',
    'Abdulkader/vrpnc12a.txt', 'Abdulkader/vrpnc12b.txt',
    'Abdulkader/vrpnc13a.txt', 'Abdulkader/vrpnc13b.txt',
    'Abdulkader/vrpnc14a.txt', 'Abdulkader/vrpnc14b.txt',
]

def check_parameters():
    parser = argparse.ArgumentParser(description="Routing optimization with solution shaking to escape local optima.")

    parser.add_argument(
        "--output-file", 
        type = str,
        help = "Path to the output file where the results of the algorithm will be saved."
    )

    parser.add_argument(
        "--max-neighborhood", 
        type = int,
        default = 20,
        help="Maximum neighborhood size to shake the solution and escape local optima (max 50)."
    )

    parser.add_argument(
        "--runs", 
        type = int,
        default = 1,
        help = "Number of times the program will be executed for each instance."
    )

    args = parser.parse_args()

    if args.max_neighborhood > 50:
        parser.error("--max-neighborhood must be less than or equal to 50.")
    
    return args


args = check_parameters()

program = "./mcvrp.py"


if args.output_file:
    with open(args.output_file, "w") as out_file:
        for file_name in file_names:
            print(f"Running instance: {file_name}")
            try:
                result = subprocess.run(
                    [python_path, program, "--instance", str(file_name), "--max-neighborhood", str(args.max_neighborhood), "--runs", str(args.runs)],
                    text=True,
                    capture_output=True
                )
                out_file.write(f"Instance: {file_name}\n")
                if result.stdout:
                    out_file.write(result.stdout)
                if result.stderr:
                    out_file.write(f"Error: {result.stderr}")
                out_file.write("\n\n")
            except Exception as e:
                out_file.write(f"Error running instance {file_name}: {e}\n\n")

else:
    for file_name in file_names:
        print(f"Running instance: {file_name}")
        try:
            result = subprocess.run(
                [python_path, program, "--instance", str(file_name), "--max-neighborhood", str(args.max_neighborhood), "--runs", str(args.runs)],
                text=True,
                capture_output=True
            )
            print(f"Instance: {file_name}")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")
            print("\n")
        except Exception as e:
            print(f"Error running instance {file_name}: {e}\n")