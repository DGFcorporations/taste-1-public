"""taste CLI — operator surface for taste-1.

    taste ingest <file.md>                — extract candidate layers
    taste standards [--state=..]          — list corpus
    taste confirm <std_id>                — activate a provisional layer
    taste packet --type=write_doc         — compile the packet for a task
    taste feedback --pick=A --why="..."   — log an owner decision
    taste queue                           — pending inferred items
    taste serve [--sock PATH]             — print the serve command
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import create_app
from .ingest import ingest_docs
from .inject import compile_packet
from .models import LayerState, PacketRequest


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="taste")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("file")
    ps = sub.add_parser("standards")
    ps.add_argument("--state", default=None)
    pc = sub.add_parser("confirm")
    pc.add_argument("std_id")
    pp = sub.add_parser("packet")
    pp.add_argument("--type", default="default")
    pf = sub.add_parser("feedback")
    pf.add_argument("--pick", required=True)
    pf.add_argument("--why", required=True)
    sub.add_parser("queue")
    pam = sub.add_parser("amend")
    pam.add_argument("--directive", required=True)
    pam.add_argument("--why", required=True)
    pam.add_argument("--polarity", default="never")
    pam.add_argument("--category", default="structure")
    pha = sub.add_parser("harvest")
    pha.add_argument("file")
    pha.add_argument("--type", default="task_run")
    pre = sub.add_parser("resolve")
    pre.add_argument("queue_id")
    pre.add_argument("--accept", action="store_true", default=False)
    pre.add_argument("--why", default=None)
    pv = sub.add_parser("serve")
    pv.add_argument("--sock", default="./taste1.sock")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        print(f"  granian --interface asgi --uds {args.sock} taste.api:app")
        print(f"  # or: uvicorn --uds {args.sock} taste.api:app")
        return 0

    app = create_app()
    engine, store = app.state.engine, app.state.store

    if args.cmd == "ingest":
        text = Path(args.file).read_text()
        resp = ingest_docs("owner", [{"id": args.file, "text": text}])
        for c in resp.candidates:
            engine.propose(c)
        print(f"{len(resp.candidates)} candidate layers (provisional):")
        for r in resp.restatements:
            print(f"  [{r.layer.polarity.value}] {r.layer.directive}")
            for imp in r.implications[:2]:
                print(f"      → {imp}")

    elif args.cmd == "standards":
        for s in store.list_standards(
                "owner", state=LayerState(args.state) if args.state else None):
            print(f"[{s.state.value:11}] [{s.polarity.value:5}] "
                  f"{s.id}  {s.directive[:70]}")

    elif args.cmd == "confirm":
        print("activated" if engine.confirm(args.std_id) else "not found")

    elif args.cmd == "packet":
        pkt = compile_packet(engine, PacketRequest(
            task_ctx={"type": args.type}))
        print(pkt.model_dump_json(indent=2))

    elif args.cmd == "feedback":
        from .models import Category, DecisionRecord, Polarity, Provenance, Standard
        rec = DecisionRecord(pick=args.pick, rationale=args.why)
        store.put_decision(rec)
        engine.propose(Standard(
            category=Category.DOCTRINE, polarity=Polarity.WHY,
            directive=f"Prefer what won as '{args.pick}'",
            rationale=args.why, provenance=Provenance.DECISION,
            variant_meta={"decision_id": rec.id}))
        print(f"decision {rec.id} logged; 1 provisional why-layer extracted")

    elif args.cmd == "amend":
        from .models import Category, Polarity
        from .triage import propose_amendment
        std = propose_amendment(
            engine, "owner", args.directive, args.why,
            polarity=Polarity(args.polarity),
            category=Category(args.category),
        )
        print(f"provisional amendment {std.id} proposed; enqueued for review")

    elif args.cmd == "harvest":
        from .harvest import harvest_task_run
        text = Path(args.file).read_text(encoding="utf-8")
        res = harvest_task_run(engine, "owner", {"type": args.type}, text)
        print(f"harvested {res.harvested} candidate(s); queued {res.queued} for review")

    elif args.cmd == "resolve":
        from .triage import resolve_amendment
        if args.queue_id.startswith("amend_"):
            res = resolve_amendment(engine, args.queue_id, accept=args.accept, rationale=args.why)
            print(f"{res['queue_id']}: {res['status']}")
        else:
            store.resolve_inferred(args.queue_id)
            print(f"{args.queue_id}: resolved")

    elif args.cmd == "queue":
        for item in store.pending_inferred("owner"):
            print(f"[{item['kind']}] {item['id']}  {item['ts']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
