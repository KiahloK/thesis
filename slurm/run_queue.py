#!/usr/bin/env python3
"""
run_queue.py  -  work through a plan of Slurm runs unattended (e.g. overnight).

A *plan* (JSON, see slurm/overnight_plan.json) lists runs. Each run is a baseline.ipynb or
refined.ipynb run over a query range, split into chunks that fit the dev partition's 30-minute
limit. The driver
  - submits the next chunk whenever Slurm accepts another job (the dev queue only allows a few
    submitted jobs per user, so jobs are fed in one by one instead of with --dependency),
  - starts a refined run only when every chunk of the baseline it builds on is done, and passes
    those chunk directories as BASELINE_DIR,
  - sizes chunks from the timing of chunks that already finished (summary.json -> timing),
  - splits a chunk in half and retries it when its job ended without output (time limit, crash),
  - keeps its state in a JSON file, so it can be stopped and restarted at any time.

Run it on the login node from the project root, inside tmux/screen so it survives logout:
    tmux new -s queue
    .venv/bin/python slurm/run_queue.py slurm/overnight_plan.json
    (detach with Ctrl-b d, reattach with: tmux attach -t queue)
Status at any time (also from another shell):
    .venv/bin/python slurm/run_queue.py slurm/overnight_plan.json --status
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = {"baseline": "slurm/baseline.sbatch", "refined": "slurm/refined.sbatch"}
JOB_LIMIT_SECONDS = 30 * 60
SAFETY = 0.75          # use at most 75 % of the time limit for a chunk
MAX_ATTEMPTS = 3       # per chunk of size 1 before it is given up
MAX_CONSECUTIVE_FAILURES = 4  # a run whose jobs keep ending without output is stopped (likely a bug)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Queue:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self.plan = json.loads(plan_path.read_text())
        self.state_path = plan_path.with_name(plan_path.stem + "_state.json")
        self.log_path = plan_path.with_name(plan_path.stem + ".log")
        self.poll = int(self.plan.get("poll_seconds", 60))
        self.state = self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
        else:
            state = {"runs": {}}
        for run in self.plan["runs"]:
            name = run["name"]
            if name in state["runs"]:
                continue
            start, end = run.get("queries", self.plan.get("queries", [0, 30]))
            if run.get("existing_dirs"):
                # Already done outside the driver (e.g. a baseline run by hand).
                state["runs"][name] = {"todo": [], "submitted": {}, "done": {str(start): {
                    "end": end, "dirs": run["existing_dirs"]}}, "failed": []}
            else:
                state["runs"][name] = {"todo": [[start, end]], "submitted": {}, "done": {}, "failed": []}
        return state

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def run_def(self, name: str) -> dict:
        return next(r for r in self.plan["runs"] if r["name"] == name)

    def finished(self, name: str) -> bool:
        s = self.state["runs"][name]
        return not s["todo"] and not s["submitted"]

    def done_dirs(self, name: str) -> list[str]:
        s = self.state["runs"][name]
        dirs = []
        for start in sorted(s["done"], key=int):
            dirs.extend(s["done"][start]["dirs"])
        return dirs

    # ------------------------------------------------------------------ slurm

    def my_jobs(self) -> set[str] | None:
        """Job ids of the user's jobs in the queue; None if squeue fails (then: don't conclude)."""
        try:
            out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                                 capture_output=True, text=True, check=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return {line.strip() for line in out.splitlines() if line.strip()}

    def submit(self, name: str, start: int, end: int) -> str | None:
        run = self.run_def(name)
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.plan.get("env", {}).items()})
        env.update({k: str(v) for k, v in run.get("env", {}).items()})
        env.update({"QUERY_OFFSET": str(start), "QUERY_LIMIT": str(end - start)})
        if run["kind"] == "refined":
            env["LOAD_BASELINE"] = "1"
            env["BASELINE_DIR"] = ",".join(self.done_dirs(run["baseline"]))
        cmd = ["sbatch", "--parsable", f"--job-name={name}-q{start}", SCRIPTS[run["kind"]]]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            return None  # usually the per-user submit limit: try again later
        return result.stdout.strip().split(";")[0]

    # ------------------------------------------------------------------ outputs

    def find_output(self, job_id: str) -> str | None:
        """The output dir written by a job (its summary.json records SLURM_JOB_ID)."""
        for summary in Path("output").glob("*/summary.json"):
            try:
                config = json.loads(summary.read_text()).get("config", {})
            except (OSError, ValueError):
                continue
            if str(config.get("slurm_job_id")) == job_id:
                return str(summary.parent)
        return None

    def seconds_per_query(self, name: str) -> float | None:
        """Pessimistic per-query time of a run from its finished chunks (max of chunk maxima)."""
        worst = None
        for entry in self.state["runs"][name]["done"].values():
            for d in entry["dirs"]:
                try:
                    timing = json.loads((Path(d) / "summary.json").read_text()).get("timing", {})
                except (OSError, ValueError):
                    continue
                stages = timing.get("per_query_seconds", {})
                # generation (baseline) or analysis + filtering/check/refinement (refined), per query
                total = sum(v["max"] for k, v in stages.items() if k in ("initial_generation", "analysis", "refinement_total"))
                if total:
                    worst = max(worst or 0, total)
        return worst

    def chunk_size(self, name: str) -> int:
        run = self.run_def(name)
        cap = self.state["runs"][name].get("cap", math.inf)  # set after a chunk failed
        measured = self.seconds_per_query(name)
        if measured:
            load = float(self.plan.get("model_load_seconds", 150))
            size = math.floor((JOB_LIMIT_SECONDS * SAFETY - load) / measured)
        else:
            size = run.get("chunk", 5)
        return int(max(1, min(run.get("max_chunk", 15), size, cap)))

    # ------------------------------------------------------------------ main loop

    def step(self) -> None:
        jobs = self.my_jobs()
        if jobs is None:
            self.log("squeue failed - retrying later")
            return
        # 1. collect finished jobs
        for name, s in self.state["runs"].items():
            for job_id, (start, end) in list(s["submitted"].items()):
                if job_id in jobs:
                    continue
                out = self.find_output(job_id)
                del s["submitted"][job_id]
                if out:
                    s["done"][str(start)] = {"end": end, "dirs": [out], "job": job_id}
                    s["consecutive_failures"] = 0
                    self.log(f"{name} q{start}-{end}: done (job {job_id}) -> {out}")
                else:
                    s["cap"] = min(s.get("cap", math.inf), max(1, (end - start) // 2))
                    s["consecutive_failures"] = s.get("consecutive_failures", 0) + 1
                    if s["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                        s["failed"].append([start, end])
                        s["failed"].extend(s["todo"])
                        s["todo"] = []
                        self.log(f"{name}: STOPPED after {s['consecutive_failures']} jobs in a row ended without output - "
                                 f"probably an error, not the time limit; see slurm-{name}-q{start}-{job_id}.out / .err")
                        self.save()
                        continue
                    attempts = s.setdefault("attempts", {}).get(f"{start}-{end}", 0) + 1
                    s["attempts"][f"{start}-{end}"] = attempts
                    if end - start > 1:
                        mid = (start + end) // 2
                        s["todo"][:0] = [[start, mid], [mid, end]]
                        self.log(f"{name} q{start}-{end}: job {job_id} ended WITHOUT output "
                                 f"(time limit or error, see slurm-{name}-q{start}-{job_id}.out) -> retry as q{start}-{mid}, q{mid}-{end}")
                    elif attempts < MAX_ATTEMPTS:
                        s["todo"].insert(0, [start, end])
                        self.log(f"{name} q{start}-{end}: job {job_id} ended WITHOUT output -> retry ({attempts}/{MAX_ATTEMPTS})")
                    else:
                        s["failed"].append([start, end])
                        self.log(f"{name} q{start}-{end}: job {job_id} ended WITHOUT output -> GIVEN UP after {attempts} attempts")
                self.save()
        # 2. submit as much as Slurm accepts, in plan order
        for run in self.plan["runs"]:
            name, s = run["name"], self.state["runs"][run["name"]]
            if run["kind"] == "refined":
                base = self.state["runs"][run["baseline"]]
                if not self.finished(run["baseline"]):
                    continue
                if base["failed"]:
                    if s["todo"]:
                        self.log(f"{name}: SKIPPED - baseline {run['baseline']} has failed chunks {base['failed']}")
                        s["failed"].extend(s["todo"])
                        s["todo"] = []
                        self.save()
                    continue
            while s["todo"]:
                start, end = s["todo"][0]
                size = min(end - start, self.chunk_size(name))
                job_id = self.submit(name, start, start + size)
                if job_id is None:
                    return  # Slurm accepts nothing more right now
                if size < end - start:
                    s["todo"][0] = [start + size, end]
                else:
                    s["todo"].pop(0)
                s["submitted"][job_id] = [start, start + size]
                self.log(f"{name} q{start}-{start + size}: submitted as job {job_id}")
                self.save()

    def all_finished(self) -> bool:
        return all(self.finished(r["name"]) for r in self.plan["runs"])

    def run_forever(self) -> None:
        self.log(f"driver started for {self.plan_path} ({len(self.plan['runs'])} runs)")
        while not self.all_finished():
            self.step()
            if self.all_finished():
                break
            time.sleep(self.poll)
        self.log("all runs finished")
        self.print_status()

    def print_status(self) -> None:
        print(f"{'run':<22} {'done':>9} {'running':>8} {'todo':>6} {'failed':>7}  output dirs")
        for run in self.plan["runs"]:
            s = self.state["runs"][run["name"]]
            n_done = sum(e["end"] - int(k) for k, e in s["done"].items())
            n_run = sum(e - b for b, e in s["submitted"].values())
            n_todo = sum(e - b for b, e in s["todo"])
            n_fail = sum(e - b for b, e in s["failed"])
            print(f"{run['name']:<22} {n_done:>9} {n_run:>8} {n_todo:>6} {n_fail:>7}  {', '.join(self.done_dirs(run['name']))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--status", action="store_true", help="print the status and exit")
    args = parser.parse_args()
    if not Path(SCRIPTS["baseline"]).exists():
        sys.exit("Run this from the project root (where slurm/ and output/ are).")
    queue = Queue(args.plan)
    if args.status:
        queue.print_status()
        return
    queue.save()
    queue.run_forever()


if __name__ == "__main__":
    main()
