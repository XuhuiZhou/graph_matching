import csv

with open('test_cite.csv','w') as correct:
    writer = csv.writer(correct, quotechar='"')
    with open('test_cite', 'r') as mycsv:
        reader = csv.reader((line.replace('\0','') for line in mycsv))
        for row in reader:
            writer.writerow(row)