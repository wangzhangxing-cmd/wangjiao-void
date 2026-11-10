# -*- coding: utf-8 -*-
"""Prepare the restart of an Abaqus job that was terminated from outside.

    abaqus python continue_job.py V5_B10_EP_medium

Run it in the folder that holds the job files.  The job chain is
JOB, JOB_R1, JOB_R2, ...; the newest member that has a ``.sta`` file is the one
to continue.  The script

* stops (writes ``DONE``) if that member completed successfully;
* otherwise finds the last converged increment in its ``.sta`` file (the
  model writes restart data after every increment, OVERLAY), and writes
  ``JOB_R<k+1>.inp`` containing ``*Restart, read, step=S, inc=I`` followed by
  the step definitions that come after step S in the original ``JOB.inp``;
* writes ``next_job.txt`` = "<new job> <old job>" for run_models.bat, which then
  runs ``abaqus job=<new job> oldjob=<old job>``.

Works with Python 2.7 and 3.  The extractor reads the whole chain and takes
every step from the member that finished it.
"""
from __future__ import print_function

import os
import re
import sys


def chain(job):
    members = [job]
    k = 1
    while os.path.isfile("%s_R%d.inp" % (job, k)):
        members.append("%s_R%d" % (job, k))
        k += 1
    return [m for m in members if os.path.isfile(m + ".sta")]


def last_converged(sta_path):
    """(step, increment) of the last converged increment in a .sta file."""
    last = None
    with open(sta_path, "rb") as handle:
        for raw in handle:
            line = raw.decode("latin-1")
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)(U?)\s+\d+\s+\d+\s+\S+", line)
            if m and not m.group(4):
                last = (int(m.group(1)), int(m.group(2)))
    return last


def completed(sta_path):
    with open(sta_path, "rb") as handle:
        return b"COMPLETED SUCCESSFULLY" in handle.read()


def diverged(job):
    """True if the job stopped on its own (minimum increment), not killed."""
    path = job + ".msg"
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as handle:
        return b"TIME INCREMENT REQUIRED IS LESS THAN THE MINIMUM" in handle.read()


RELAXED = ("*Controls, parameters=field, field=displacement\n"
           "0.01, 1.0\n")


def relax(block, suffix=""):
    """Looser equilibrium tolerances (R_n 0.005 -> 0.01, C_n 0.1 -> 1.0)."""
    block = re.sub(r"(?im)^\*Controls, parameters=field, field=displacement\s*\n[^\n]*\n", RELAXED, block)
    if suffix:
        block = re.sub(r"(?i)(\*Step, name=)([^,\s]+)", lambda m: m.group(1) + re.sub(r"_C\d+$", "", m.group(2)) + suffix, block, count=1)
        block = re.sub(r"(?im)^(\*Static\s*\n)[^\n]*\n", r"\g<1>0.01, 1., 1e-10, 0.05\n", block)
    return block


def step_blocks(inp_path):
    """Original step definitions in order: list of text blocks."""
    with open(inp_path) as handle:
        text = handle.read()
    starts = [m.start() for m in re.finditer(r"(?im)^\*Step\b", text)]
    blocks = []
    for k, start in enumerate(starts):
        end = text.lower().find("*end step", start)
        end = text.find("\n", end) + 1
        blocks.append(text[start:end])
    return blocks


def timeline(members):
    """Step definitions in analysis order for the chain up to members[-1].

    A restart member R_k that reads step s continues with its own steps as
    steps s+1, s+2, ... (with END STEP, step s is closed and the next one is a
    new step), so the history is history[:s] + steps of R_k.
    """
    steps = step_blocks(members[0] + ".inp")
    for m in members[1:]:
        text = open(m + ".inp").read()
        r = re.search(r"(?im)^\*Restart,\s*read,\s*step=(\d+)", text)
        steps = steps[:int(r.group(1))] + step_blocks(m + ".inp")
    return steps


def main():
    if len(sys.argv) < 2:
        print("usage: abaqus python continue_job.py JOB")
        sys.exit(2)
    job = sys.argv[-1]
    members = chain(job)
    if not members:
        raise SystemExit("No .sta file for %s; run the job first" % job)
    current = members[-1]
    if completed(current + ".sta"):
        open("next_job.txt", "w").write("DONE %s\n" % current)
        print("%s completed successfully; nothing to continue" % current)
        return
    stuck = diverged(current)
    # A restart that failed before converging any increment has no restart
    # point of its own: go back to the newest member that has one.
    point = None
    while members:
        current = members[-1]
        point = last_converged(current + ".sta")
        if point is not None:
            break
        members.pop()
    if point is None:
        raise SystemExit("no converged increment in any .sta of %s; rerun %s" % (job, job))
    step, inc = point
    blocks = timeline(members)
    if step > len(blocks):
        raise SystemExit("Step %d not found in the step history of %s" % (step, current))
    k = 1
    while os.path.isfile("%s_R%d.sta" % (job, k)) or os.path.isfile("%s_R%d.inp" % (job, k)):
        k += 1
    new = "%s_R%d" % (job, k)
    with open(new + ".inp", "w") as out:
        out.write("*Heading\n** restart of %s from step %d increment %d\n" % (current, step, inc))
        if stuck:
            # The step itself did not converge (not an external kill): end it at
            # the last converged increment, finish its load in a new step
            # <name>_C<k>, and run all later steps with looser tolerances.
            out.write("*Restart, read, step=%d, inc=%d, end step\n" % (step, inc))
            out.write(relax(blocks[step - 1], "_C%d" % k))
            for block in blocks[step:]:
                out.write(relax(block))
        else:
            out.write("*Restart, read, step=%d, inc=%d\n" % (step, inc))
            for block in blocks[step:]:          # steps after the one being continued
                out.write(block)
    open("next_job.txt", "w").write("%s %s\n" % (new, current))
    print("wrote %s.inp: continue %s from step %d increment %d, then %d more steps%s"
          % (new, current, step, inc, len(blocks) - step,
             " (step did not converge: relaxed tolerances)" if stuck else ""))


main()
