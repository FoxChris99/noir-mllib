import csv

def swap_columns(csv_file, col_index1, col_index2):
    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Validate column indices
    num_columns = len(rows[0])
    if col_index1 < 0 or col_index1 >= num_columns or col_index2 < 0 or col_index2 >= num_columns:
        print("Invalid column indices!")
        return

    # Swap the columns
    for row in rows:
        row[col_index1], row[col_index2] = row[col_index2], row[col_index1]

    # Write the modified data back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Columns swapped successfully!")

# Prompt the user for input
csv_file = input("Enter the CSV file name: ")
col_index1 = int(input("Enter the first column index to swap (starting from 0): "))
col_index2 = int(input("Enter the second column index to swap (starting from 0): "))

# Call the function to swap columns
swap_columns(csv_file, col_index1, col_index2)

