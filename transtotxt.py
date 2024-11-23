import csv
import os

# set the file path and folder
input_csv = "MBTI 500.csv"
output_folder = "txt_output"

# make the file and check
os.makedirs( output_folder , exist_ok = True )

# read csv and approach rows
with open( input_csv , mode = 'r' , encoding='utf-8' ) as csv_file :
    reader = csv.reader(csv_file)
    header = next(reader)
    
    for i , row in enumerate(reader):
        output_file = os.path.join(output_folder , f"{i+1}.txt")
        with open(output_file , mode = 'w' , encoding='utf-8') as file:
            file.write( ",".join(row) )
        print(f"writed {output_file}")
print("success")
