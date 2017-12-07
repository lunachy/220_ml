# -*- coding: utf-8 -*-
import csv
import os

root_dir = "/data/root/pe_classify/"
upx = os.path.join(root_dir, "upx")
aspack = os.path.join(root_dir, "aspack")
csv_path = os.path.join(root_dir, "peid_result.csv")
pefile_path = os.path.join(root_dir, "all_pefile")
out_folder = os.path.join(root_dir, "unpack_pefile")


def unpack_all(pack_path):
    with open(pack_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ASPack" in row["packed"]:
                unpack(upx, row['md5'])
            elif "UPX" in row["packed"]:
                unpack(aspack, row['md5'])


def unpack(pack, md5):
    file_path = os.path.join(pefile_path, md5)
    unpack_path = os.path.join(out_folder, md5)
    os.system("%s -d %s -o %s" % (pack, file_path, unpack_path))


if __name__ == "__main__":
    unpack_all(csv_path)
