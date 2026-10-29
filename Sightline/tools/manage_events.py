r"""Command-line access to the manual event log (same store the app uses).

    python tools\manage_events.py list
    python tools\manage_events.py check
    python tools\manage_events.py add --video data\video_006.avi --frame 2384 --label Partial --notes "re-entered by me"
    python tools\manage_events.py import path\to\old_events.json

`add` creates the event and (unless --no-evidence) saves the exact frame as its evidence image. Nothing is ever
replaced: new events get new ids, and every destructive change makes a backup in outputs\backups first.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.events import VISIBILITY_LABELS  # noqa: E402
from sightline.paths import OUTPUTS, resolve  # noqa: E402
from sightline.store import EventStore, EventStoreError  # noqa: E402
from sightline.video import VideoError, VideoSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    a_ = sub.add_parser("add")
    a_.add_argument("--video", required=True)
    a_.add_argument("--frame", type=int, required=True)
    a_.add_argument("--label", default="Unassessed", choices=VISIBILITY_LABELS)
    a_.add_argument("--notes", default="")
    a_.add_argument("--type", default="pedestrian-vehicle interaction")
    a_.add_argument("--no-evidence", action="store_true")
    i_ = sub.add_parser("import")
    i_.add_argument("file")
    a = ap.parse_args()

    store = EventStore(OUTPUTS)
    if store.blocked:
        sys.exit(store.blocked)
    try:
        if a.cmd == "list":
            for e in store.events:
                print(f"#{e.event_id}  frame {e.frame}  {e.time_s:.3f}s  {e.visibility_label}  {e.video}  "
                      f"{e.notes[:50]!r}  evidence={e.evidence_frame or '-'}")
            print(f"{len(store.events)} event(s) in {store.json_path} (next id {store.next_id})")
        elif a.cmd == "check":
            c = store.consistency()
            print(c)
            sys.exit(0 if c["ok"] else 1)
        elif a.cmd == "add":
            vid = VideoSource(resolve(a.video))
            writer = None if a.no_evidence else (lambda dest: vid.save_frame(a.frame, dest))
            ev, res = store.add(a.frame, vid.fps, video=Path(a.video).name, evidence_writer=writer,
                                event_type=a.type, visibility_label=a.label, notes=a.notes)
            print(f"Saved event #{ev.event_id} (frame {ev.frame}, {ev.time_s:.3f}s) in {res.json_path}")
            if res.csv_error:
                print("WARNING:", res.csv_error)
        elif a.cmd == "import":
            added, skipped, _ = store.import_file(resolve(a.file))
            print(f"Imported {len(added)} event(s); skipped {skipped} already present.")
    except (EventStoreError, VideoError) as e:
        sys.exit(f"Not done: {e}")


if __name__ == "__main__":
    main()
