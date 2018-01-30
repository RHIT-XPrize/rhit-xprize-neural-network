import csv

input_file_name = './test.csv'
output_neural_in = './neural_in.csv'
output_neural_out = './neural_out.csv'

with open(input_file_name, 'r') as csv_in_file:
    with open(output_neural_in, 'w') as csv_neural_in_file:
        with open(output_neural_out, 'w') as csv_neural_out_file:
            reader = csv.reader(csv_in_file)
            neural_in_writer = csv.writer(csv_neural_in_file)
            neural_out_writer = csv.writer(csv_neural_out_file)

            for row in reader:
                input_cols = row[:14] + row[17:]
                output_cols = row[14:17]

                neural_in_writer.writerow(input_cols)
                neural_out_writer.writerow(output_cols)
