import argparse
import os
import os.path as osp

import UtilsBase
from UtilsBase import twrite, tqdm

P = argparse.ArgumentParser()
P.add_argument("--folder", type=str, default=os.getcwd())
args = P.parse_args()

args.folder = osp.abspath(args.folder)

can_remove = []
folder2missing_files = {}
for fpath in tqdm(list(os.listdir(args.folder))):
    if not osp.isdir(osp.join(args.folder, fpath)):
        continue

    elif osp.exists(osp.join(f"/NAS/IMLE-SSL/{osp.basename(args.folder)}", fpath)):
        local_files = os.listdir(osp.join(args.folder, fpath))
        nas_files = os.listdir(osp.join(f"/NAS/IMLE-SSL/{osp.basename(args.folder)}", fpath))

        # Find extra files here and files that aren't in the NAS.
        # If the NAS has extra files, then we can remove the local ones too.
        missing_files = [f for f in local_files if f not in nas_files]
        extra_files = [f for f in nas_files if f not in local_files]
        if len(missing_files) == 0:
            can_remove.append(osp.join(args.folder, fpath))
            folder2missing_files[fpath] = missing_files

    # elif len([f for f in os.listdir(osp.join(args.folder, fpath)) if f.endswith(".pt") and not f.startswith("wandb_data") and not f.endswith("latest.pt")]) == 0:

    #     can_remove.append(osp.join(args.folder, fpath))

    else:
        pass


        





    # # for fpath2 in os.listdir(osp.join(args.folder, fpath)):
    # #     if fpath2.endswith("_latest.pt"):

    # #         checkpoint_number = UtilsBase.remove_nonnumeric(fpath2)
    # #         checkpoint_number = int(checkpoint_number)
    # #         next_checkpoint_number = checkpoint_number + 1


    # #         if osp.exists(osp.join(args.folder, fpath, f"{next_checkpoint_number}.pt")):
    # #             can_remove.append(osp.join(args.folder, fpath, fpath2))

    # for fpath2 in os.listdir(osp.join(args.folder, fpath)):
    #     if fpath2.endswith("_latest.pt"):

    #         for fpath3 in os.listdir(osp.join(args.folder, fpath)):
    #             if fpath3.endswith(".pt") and not fpath3.startswith("wandb_data"):
    #                 fpath3_number = UtilsBase.remove_nonnumeric(fpath3)
    #                 if not fpath3_number.isdigit():
    #                     continue
    #                 else:
    #                     fpath3_number = int(fpath3_number)
                    
    #                 fpath2_number = UtilsBase.remove_nonnumeric(fpath2)
    #                 if not fpath2_number:
    #                     continue
    #                 else:
    #                     fpath2_number = int(fpath2_number)

    #                 if fpath3_number > fpath2_number:
    #                     can_remove.append(osp.join(args.folder, fpath, fpath2))
    #                     break


twrite(can_remove=can_remove, num_can_remove=len(can_remove))


import shutil
for c in tqdm(can_remove):
    print(c)
    
    for f in os.listdir(c):
        print(f"\t{f}")


    shutil.rmtree(c)
