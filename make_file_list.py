import os
from argparse import ArgumentParser

from utils.file import list_image_files


parser = ArgumentParser()
parser.add_argument("--train_folder", type=str, required=True)
parser.add_argument("--valid_folder", type=str, required=True)
parser.add_argument("--save_folder", type=str, required=True)
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

train_files = list_image_files(
    args.train_folder, exts=(".jpg", ".png", ".jpeg"), follow_links=args.follow_links,
    log_progress=True, log_every_n_files=10000
)

valid_files = list_image_files(
    args.valid_folder, exts=(".jpg", ".png", ".jpeg"), follow_links=args.follow_links,
    log_progress=True, log_every_n_files=10000
)

print(f"find {len(train_files)} images in {args.train_folder}")
print(f"find {len(valid_files)} images in {args.valid_folder}")


os.makedirs(args.save_folder, exist_ok=True)

with open(os.path.join(args.save_folder, "train.list"), "w") as fp:
    for file_path in train_files:
        fp.write(f"{file_path}\n")

valid_files = sorted(valid_files)
with open(os.path.join(args.save_folder, "valid.list"), "w") as fp:
    for file_path in valid_files:
        fp.write(f"{file_path}\n")
