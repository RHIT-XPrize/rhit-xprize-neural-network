import csv
import sys

def split_file(num_blocks, input_file_name, output_neural_in, output_neural_out):
    with open(input_file_name, 'r') as csv_in_file, \
         open(output_neural_in, 'w') as csv_neural_in_file, \
         open(output_neural_out, 'w') as csv_neural_out_file:
        reader = csv.reader(csv_in_file)
        neural_in_writer = csv.writer(csv_neural_in_file)
        neural_out_writer = csv.writer(csv_neural_out_file)

        # remove headers
        row = reader.__next__()

        for row in reader:
            state_cols = row[:num_blocks * 7]
            word_col = row[-3]
            input_cols = state_cols + [word_col]
            output_cols = [0] * num_blocks
            output_cols[int(row[-6]) - 1] = 1

            neural_in_writer.writerow(input_cols)
            neural_out_writer.writerow(output_cols)

def main():
    if len(sys.argv) != 5:
        print('Must call as "python split_file.py <num-blocks> <in-file> <neural-in-file> <neural-out-file>"')
        return

    try: 
        num_blocks = int(sys.argv[1])
    except:
        print('<num-blocks> must be an integer')
        return

    in_file = sys.argv[2]
    neural_in_file = sys.argv[3]
    neural_out_file = sys.argv[4]

    split_file(num_blocks, in_file, neural_in_file, neural_out_file)

if __name__ == '__main__':
    main()
