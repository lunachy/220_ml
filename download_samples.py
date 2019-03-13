#!/usr/bin/env python

import os
import csv
# import Queue
import zipfile
import requests
import argparse
import multiprocessing

# TODO: Don't hardcode the relative path?
samples_path = "gym_malware/envs/utils/samples/"
hashes_path = "gym_malware/envs/utils/sample_hashes.csv"
vturl = "https://www.virustotal.com/intelligence/download"


def get_sample_hashes():
    hash_rows = []
    with open(hashes_path) as csvfile:
        for row in csv.DictReader(csvfile):
            hash_rows.append(row)
    return hash_rows


def vt_download_sample(sha256, sample_path, vtapikey):
    tries = 0
    success = False
    while not success and tries < 10:
        resp = requests.get(vturl, params={"hash": sha256, "apikey": vtapikey})

        if not resp.ok:
            tries += 1
            continue

        else:
            success = True

    if not success:
        return False

    with open(sample_path, "wb") as ofile:
        ofile.write(resp.content)

    return True


def download_worker_function(download_queue, vtapikey):
    while True:
        try:
            sha256 = download_queue.get()
        except queue.Empty:
            continue

        if sha256 == "STOP":
            download_queue.task_done()
            return True

        print("{} downloading".format(sha256))
        sample_path = os.path.join(samples_path, sha256)
        success = vt_download_sample(sha256, sample_path, vtapikey)

        if not success:
            print("{} had a problem".format(sha256))

        print("{} done".format(sha256))
        download_queue.task_done()


def use_virustotal(args):
    """
    Use Virustotal to download the environment malware
    """
    m = multiprocessing.Manager()
    download_queue = m.JoinableQueue(args.nconcurrent)

    archive_procs = [
        multiprocessing.Process(
            target=download_worker_function,
            args=(download_queue, args.vtapikey))
        for i in range(args.nconcurrent)
    ]
    for w in archive_procs:
        w.start()

    for row in get_sample_hashes():
        download_queue.put(row["sha256"])

    for i in range(args.narchiveprocs):
        download_queue.put("STOP")

    download_queue.join()
    for w in archive_procs:
        w.join()


def use_virusshare(args):
    """
    Use VirusShare zip files as the source for the envirnment malware
    """
    pwd = bytes(args.zipfilepassword, "ascii")
    md5_to_sha256_dict = {d["md5"]: d["sha256"] for d in get_sample_hashes()}

    for path in args.zipfile:
        z = zipfile.ZipFile(path)
        for f in z.namelist():
            z_object_md5 = f.split("_")[1]
            if z_object_md5 in md5_to_sha256_dict:
                sample_bytez = z.open(f, "r", pwd).read()
                with open(md5_to_sha256_dict[z_object_md5], "wb") as ofile:
                    ofile.write(sample_bytez)
                print("Extracted {}".format(md5_to_sha256_dict[z_object_md5]))


if __name__ == '__main__':
    prog = "download_samples"
    descr = "Download the samples that define the malware gym environment"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument(
        "--virustotal",
        default=False,
        action="store_true",
        help="Use Virustotal to download malware samples")
    parser.add_argument(
        "--vtapikey", type=str, default=None, help="Virustotal API key")
    parser.add_argument(
        "--nconcurrent",
        type=int,
        default=6,
        help="Maximum concurrent downloads from Virustotal")
    parser.add_argument(
        "--virusshare",
        default=False,
        action="store_true",
        help="Use malware samples from VirusShare torrents")
    parser.add_argument(
        "--zipfile",
        type=str,
        nargs="+",
        help="The path of VirusShare zipfile 290 or 291")
    parser.add_argument(
        "--zipfilepassword",
        type=str,
        default=None,
        help="Password for the VirusShare zipfiles 290 or 291")
    args = parser.parse_args()

    if not args.virustotal and not args.virusshare:
        parser.error("Must use either Virustotal or VirusShare")

    if args.virusshare:
        if len(args.zipfile) == 0:
            parser.error("Must the paths for one or more Virusshare zip files")

        if args.zipfilepassword is None:
            parser.error("Must enter a password for the VirusShare zip files")

        use_virusshare(args)

    if args.virustotal:
        if args.vtapikey is None:
            parser.error("Must enter a VirusTotal API key")

        use_virustotal(args)