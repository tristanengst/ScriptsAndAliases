"""Utility functions you might consider using in your own codebase in order for
ScriptsAndAliases to work best for displaying info about your jobs. They may require
your repo to also have UtilsBase.py available.
"""

######################################################################################
# CODE FOR HEARTBEAT FILE WRITING
# You will need to also use UtilsBase.py for this to work
# NOTE: as of 2026-02-04, this code is copied from another repo and not tested here.
######################################################################################
from datetime import datetime, timedelta
import os.path as osp
import time

import UtilsBase
from UtilsBase import twrite

def write_heartbeat_(*, exp_folder, interval=300, verbose=1):
    """Essentially the inner function for write_heartbeat, defined in global scope so
    it can be run in a separate process/thread.
    """
    hearbeat_time, last_write_time,interval_exceeded = datetime.now(), None, True
    heartbeat_file = osp.join(exp_folder, "heartbeat.txt")
    if osp.exists(heartbeat_file):
        content = UtilsBase.load_file_lite(heartbeat_file).strip()
        last_write_time = datetime.fromisoformat(content) if content else datetime.min
        interval_exceeded = (hearbeat_time - last_write_time) > timedelta(seconds=interval)
    
    if verbose >= 2:
        twrite(f"[INFO] write_heartbeat_(): hearbeat_time={hearbeat_time}, heartbeat_file={heartbeat_file} (exists={osp.exists(heartbeat_file)}, last_write_time={last_write_time}), interval_exceeded={interval_exceeded} -> {'write heartbeat' if interval_exceeded else 'skip write heartbeat'}")

    if interval_exceeded:
        time_str = hearbeat_time.isoformat(sep=' ', timespec='seconds') + "\n"
        _ = UtilsBase.atomic_save_lite(data=time_str, fpath=heartbeat_file)

def write_heartbeat(*, exp_folder, interval=300, verbose=1):
    """Writes to a heartbeat.txt file in [exp_folder] if the last heartbeat written
    doesn't exist or is older than [interval] seconds.
    """
    # Since this function can get called often and writes to not-always-stable
    # filesystems, better to ignore errors than crash
    start_time = time.time()
    try:
        result = UtilsBase.run_in_new_thread(write_heartbeat_, exp_folder=exp_folder, interval=interval, verbose=verbose)
        elapsed_time = UtilsBase.seconds_since_time(start_time)
        if verbose and elapsed_time > 5:
            twrite(f"[WARNING] write_heartbeat(): exp_folder={exp_folder} elapsed_time={elapsed_time:.2f}S")
    except Exception as e:
        twrite(f"[ERROR] write_heartbeat(): Failed to write heartbeat to {exp_folder}, elapsed_time={UtilsBase.hours_since_time(start_time)}H with error {e}")

######################################################################################
######################################################################################
######################################################################################

######################################################################################
# Code for adding commenting SLURM jobs with JSON metadata for advanced functionality
######################################################################################
def get_sbatch_comment(*, exp_name, uid):
    """Returns a string suitable for use as the --comment argument to sbatch,
    containing JSON metadata with [exp_name] and [uid], which is important for various
    ScriptsAndAliases advanced functionality.

    --- EXPECTED USAGE ---------------------------------------------------------------
    Imagine you're building SLURM scripts from templates. The template has a line:

    #SBATCH --comment=COMMENT_PLACEHOLDER

    and during the process for generating a script, you have Python code that's read
    the template into a variable [sbatch_template].

    When generating the SLURM script for a particular experiment, you would replace
    COMMENT_PLACEHOLDER with the output of this function, eg.

    sbatch_comment = get_sbatch_comment(exp_name="name_of_experiment_abcd1245", uid="abcd1234")
    sbatch_script = sbatch_template.replace("COMMENT_PLACEHOLDER", sbatch_comment)
    ----------------------------------------------------------------------------------

    Args:
    exp_name    -- name of the experiment the job is for
    uid         -- unique identifier for the job (e.g., uuid4 string)
    """
    comment = dict(uid=uid, exp_name=exp_name)

    # If [comment] is too long, replace the [exp_name] key with [exp_name_trunc] and 
    # truncate its value so everything fits in 255 characters.
    comment_str = json.dumps(comment)
    if len(comment_str) > 255:
        comment = dict(uid=uid, exp_name_trunc="")
        chars_for_exp_name = 255 - len(json.dumps(comment))
        exp_name_trunc = exp_name[:chars_for_exp_name]
        shortened_comment = dict(uid=uid, exp_name_trunc=exp_name_trunc)
        comment_str = json.dumps(shortened_comment)

    comment_str = comment_str.replace("\"", "'")
    return "\"" + comment_str + "\""

######################################################################################
######################################################################################
######################################################################################


######################################################################################
# CODE FOR UID GENERATION
######################################################################################
import uuid
def generate_uid():
    """Generates a unique identifier string."""
    return str(uuid.uuid4())[:8]
    # If you have WandB imported: return wandb.util.generate_id() also works

######################################################################################
######################################################################################
######################################################################################

