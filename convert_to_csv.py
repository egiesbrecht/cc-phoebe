#!/bin/python

import pathlib
import csv


def main(dir_path, out_file):
    with open(out_file, "w", newline="") as o:
        pathlist = pathlib.Path(dir_path).glob("**/*.txt")
        writer = csv.writer(o)
        num_labels = 10
        writer.writerow(["head", "body", "text", "strlabels"] + ["a" + str(i) for i in range(num_labels)] + ["b" + str(i) for i in range(num_labels)])
        for p in pathlist:
            with open(str(p), "r", encoding="cp1252") as f:
                c_str = f.read()
            c_parts = c_str.split("\n")
            strlabels = c_parts[-1]
            abrct_head = c_parts[0]
            abrct_body = c_parts[1]
            text = abrct_head + " [SEP] " + abrct_body

            main_label = [int(n[0]) for n in strlabels.split(",")]
            wtr_label = [0] * num_labels
            for n in main_label:
                wtr_label[n] = 1
            wsr_label = [0] * num_labels
            side_label = [(int(n[2]) if len(n) >= 2 else None) for n in strlabels.split(",")]
            for n in side_label:
                if n is not None:
                    wsr_label[n] = 1

            writer.writerow([abrct_head, abrct_body, text, strlabels] + wtr_label + wsr_label)
    print("finished")
    
if __name__ == "__main__":
    main("abstracts_new/train/", "wordpiece_abstracts_train_side_label_1.csv")
    #main("abstracts_new/test/", "wordpiece_abstracts_test.csv")

