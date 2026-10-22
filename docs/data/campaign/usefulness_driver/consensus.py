"""Driver for the XYZ^2 usefulness consensus, rounds 5 and later.

State lives in state.json. Each workflow run does one stage and skips every agent
whose result is already in a journal (matched by label), so a run cut off by a usage
limit loses only the agents that did not finish.

  python3 consensus.py fix        -> writes run_fix.js for the current round
  python3 consensus.py merge-fix  -> applies the finished fixes to the draft
  python3 consensus.py review     -> writes run_review.js (overall rewrite + next round's reviews + triage)
  python3 consensus.py merge-review -> computes agreement, decides consensus, prepares the fix tasks
  python3 consensus.py write      -> writes run_write.js (final document + verified doc fixes)
  python3 consensus.py status
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = "/global/homes/s/sohan100/.claude/projects/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/subagents/workflows"
OLD_RUN = "wf_806a106f-964"
STATE = os.path.join(HERE, "state.json")
LENSES = ["numbers: every number in the revised claim against the cited files",
          "scope and significance: the verdict, the significance statements, and whether the claim is stronger or weaker than the data"]


def load():
    return json.load(open(STATE))


def save(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE)


def labels(st):
    """label -> latest result over the old run and every run of this driver, in launch order."""
    out = {}
    runs = [OLD_RUN] + st.get("runs", [])
    for run in runs:
        path = os.path.join(WF, run, "journal.jsonl")
        if not os.path.exists(path):
            continue
        lab = {}
        for line in open(path):
            d = json.loads(line)
            if d["type"] == "started":
                lab[d["key"]] = d["label"]
            elif d["type"] == "result":
                v = d.get("value", d.get("result"))
                if v is not None and d["key"] in lab:
                    out[lab[d["key"]]] = v
    return out


def material(triage):
    return [x for x in (triage or {}).get("items", []) if x["scope"] == "substantive" and x["material"]]


def summarize(draft, reviews):
    ids = [c["id"] for c in draft["claims"]]

    def votes(cid):
        out = []
        for i, r in enumerate(reviews):
            v = next((x for x in r["reviews"] if x["claim_id"] == cid), None)
            if v:
                out.append((i, v))
        return out
    af = {}
    for cid in ids:
        vs = votes(cid)
        af[cid] = sum(v["agree"] for _, v in vs) / len(vs) if vs else 0
    ov = votes("OVERALL")
    af["OVERALL"] = (sum(v["agree"] for _, v in ov) / len(ov)) if ov else sum(r["overall_agree"] for r in reviews) / len(reviews)
    notes = {cid: [{"reviewer": i + 1, "agree": v["agree"], "severity": v["severity"], "correction": v["correction"], "evidence": v["evidence"]}
                   for i, v in votes(cid) if (not v["agree"]) or v["severity"] != "none"] for cid in ids + ["OVERALL"]}
    missing = [f"[reviewer {i + 1}] {m}" for i, r in enumerate(reviews) for m in (r.get("missing") or [])]
    doc = [x for r in reviews for x in (r.get("doc_fixes") or [])]
    return {"agreeFrac": af, "notes": notes, "missing": missing, "docNotes": doc,
            "missingCount": sum(1 for r in reviews if r.get("missing")), "reviewers": len(reviews)}


def index(draft):
    return "\n".join(f"{c['id']}: {c['regime']}" for c in draft["claims"])


def fix_tasks(st):
    """Tasks for the current round's fix stage, in claim order, plus new claims for unassigned material omissions."""
    R, draft, sm = st["round"], st["draft"], st["summary"]
    af, mat = sm["agreeFrac"], material(st["triage"])
    ids = [c["id"] for c in draft["claims"]]
    light_ok = R >= st["lightFromRound"]
    tasks = []
    for c in draft["claims"]:
        om = [m["text"] for m in mat if c["id"] in (m.get("claim_ids") or [])]
        if af[c["id"]] < 1 or om:
            tasks.append({"kind": "fix", "id": c["id"], "claim": c, "frac": af[c["id"]], "notes": sm["notes"].get(c["id"], []),
                          "omissions": om, "light": bool(light_ok and af[c["id"]] >= 0.8 and not om)})
    nxt = max(int(re.sub(r"\D", "", i) or 0) for i in ids) + 1
    for m in mat:
        if not [i for i in (m.get("claim_ids") or []) if i in ids]:
            tasks.append({"kind": "new", "id": f"C{nxt}", "text": m["text"], "reason": m["reason"]})
            nxt += 1
    return tasks


def outcome(R, t, L):
    """Replay the fix pipeline for one task from saved results: ('done', result) or ('pending', None)."""
    p = f"r{R}:{t['id']}"
    if t["kind"] == "new":
        n = L.get(f"new:{p}")
        if n is None:
            return "pending", None
        if not n.get("keep") or not n.get("claim"):
            return "done", None
        vs = [L.get(f"verify:{p}:{k}") for k in range(2)]
        if any(v is None for v in vs):
            return "pending", None
        return "done", ({**n["claim"], "id": t["id"]} if all(v["accept"] for v in vs) else None)
    f = L.get(f"fix:{p}")
    if f is None:
        return "pending", None
    if t["light"] and not f.get("drop") and not (f.get("changes") or []):
        return "done", f
    vs = [L.get(f"verify:{p}:{k}") for k in range(1 if t["light"] else 2)]
    if any(v is None for v in vs):
        return "pending", None
    probs = [q for v in vs if not v["accept"] for q in v["problems"]]
    if not probs:
        return "done", f
    g = L.get(f"refix:{p}")
    return ("done", g) if g is not None else ("pending", None)


def script(template, data, out):
    src = open(os.path.join(HERE, template)).read().replace("__INPUT__", json.dumps(data))
    open(os.path.join(HERE, out), "w").write(src)
    print(f"wrote {out}: {len(src)} chars")


def cmd_fix():
    st = load()
    L = labels(st)
    R = st["round"]
    tasks = fix_tasks(st)
    pend = [t for t in tasks if outcome(R, t, L)[0] == "pending"]
    keys = [k for t in pend for k in (f"fix:r{R}:{t['id']}", f"new:r{R}:{t['id']}", f"verify:r{R}:{t['id']}:0", f"verify:r{R}:{t['id']}:1")]
    cached = {k: L[k] for k in keys if k in L}
    print(f"round {R}: {len(tasks)} tasks, {len(pend)} pending, {len(cached)} cached results reused")
    script("tpl_fix.js", {"round": R, "brief2": st["brief2"], "rules2": st["rules2"], "claimSchema": st["claimSchema"],
                          "index": index(st["draft"]), "tasks": pend, "cached": cached, "lenses": LENSES}, "run_fix.js")


def cmd_merge_fix():
    st = load()
    L = labels(st)
    R = st["round"]
    tasks = fix_tasks(st)
    res = [(t, *outcome(R, t, L)) for t in tasks]
    pend = [t["id"] for t, s, _ in res if s == "pending"]
    if pend:
        print("still pending:", pend)
        return
    by, dropped, lines, new = {}, set(), [], []
    for t, _, f in res:
        if t["kind"] == "new":
            if f:
                new.append(f)
                lines.append(f"{f['id']}: new claim ({f['regime']})")
            continue
        if f.get("drop"):
            dropped.add(t["id"])
            lines.append(f"{t['id']}: dropped. " + " | ".join(f.get("changes") or []))
            continue
        by[t["id"]] = {**f["claim"], "id": t["id"]}
        reb = (" | rebuttals: " + " | ".join(f["rebuttals"])) if f.get("rebuttals") else ""
        lines.append(f"{t['id']}: " + (" | ".join(f.get("changes") or []) or "no change") + reb)
    d = st["draft"]
    d["claims"] = [by.get(c["id"], c) for c in d["claims"] if c["id"] not in dropped] + new
    lines.append("OVERALL: rewritten from the revised claims")
    st["changelog"] = "\n".join(lines)
    st["overallNotes"] = st["summary"]["notes"].get("OVERALL", [])
    st["stage"] = "review"
    st["overallRound"] = R
    st["round"] = R + 1
    st["summary"], st["triage"] = None, None
    save(st)
    print(f"round {R}: {len(by)} revised, {len(dropped)} dropped, {len(new)} added; next: overall r{R} + review r{R + 1}")


def cmd_review():
    st = load()
    L = labels(st)
    R = st["round"]
    N = st.get("overallRound")
    keys = [f"overall:r{N}", f"overall-verify:r{N}:0", f"overall-verify:r{N}:1", f"overall-refix:r{N}"] + [f"review:r{R}:{i}" for i in range(len(st["reviewers"]))] + [f"triage:r{R}"]
    cached = {k: L[k] for k in keys if k in L}
    print(f"overall r{st.get('overallRound')} + review r{R}: {len(cached)} cached results reused")
    script("tpl_review.js", {"round": R, "overallRound": N, "verifyOverall": bool(N and N >= st.get("verifyOverallFrom", 6)), "draft": st["draft"], "changelog": st["changelog"],
                             "overallNotes": st.get("overallNotes", []), "brief2": st["brief2"], "rules2": st["rules2"],
                             "reviewers": st["reviewers"], "cached": cached}, "run_review.js")


def cmd_merge_review():
    st = load()
    L = labels(st)
    R = st["round"]
    N = st.get("overallRound")
    ov = L.get(f"overall:r{N}") if N else None
    if N and ov is None:
        print("overall pending")
        return
    if N and N >= st.get("verifyOverallFrom", 6):
        vs = [L.get(f"overall-verify:r{N}:{k}") for k in range(2)]
        if any(v is None for v in vs):
            print("overall verification pending")
            return
        if any(not v["accept"] for v in vs):
            ov = L.get(f"overall-refix:r{N}")
            if ov is None:
                print("overall refix pending")
                return
    reviews = [L.get(f"review:r{R}:{i}") for i in range(len(st["reviewers"]))]
    if any(r is None for r in reviews):
        print("reviews pending:", [i for i, r in enumerate(reviews) if r is None])
        return
    if ov:
        st["draft"] = {**st["draft"], "overall": ov["overall"], "open_questions": ov["open_questions"]}
    sm = summarize(st["draft"], reviews)
    tri = L.get(f"triage:r{R}") if sm["missing"] else {"items": []}
    if tri is None:
        print("triage pending")
        return
    mat = material(tri)
    weak = [k for k, v in sm["agreeFrac"].items() if v < 0.8]
    st["docNotes"] = st.get("docNotes", []) + sm["docNotes"] + [x["text"] for x in tri["items"] if x["scope"] == "doc"]
    st["history"].append({"round": R, "agreeFrac": sm["agreeFrac"], "weak": weak, "missingCount": sm["missingCount"],
                          "material": len(mat), "reviewers": sm["reviewers"]})
    st["summary"], st["triage"] = sm, tri
    st["overallRound"] = None
    n = len(sm["agreeFrac"])
    print(f"round {R}: {n - len(weak)}/{n} items at >=80% agreement; weak {weak}; {len(mat)} material omissions")
    if not weak and not mat:
        st["consensus"], st["stage"] = True, "write"
    elif R >= st["maxRound"]:
        st["consensus"], st["stage"] = False, "write"
    else:
        st["stage"] = "fix"
    save(st)
    print("next stage:", st["stage"])


def cmd_write():
    st = load()
    L = labels(st)
    keys = ["write:final", "docfix:merge"] + [k for k in L if k.startswith("docfix:") and k != "docfix:merge"]
    cached = {k: L[k] for k in keys if k in L}
    last = st["history"][-1]["agreeFrac"]
    residual = {cid: [n for n in st["summary"]["notes"].get(cid, []) if not n["agree"]] for cid, v in last.items() if v < 0.8}
    omissions = [x for x in (st.get("triage") or {}).get("items", []) if x["scope"] == "substantive" and x["material"]]
    script("tpl_write.js", {"residual": residual, "omissions": omissions,
                            "draft": st["draft"], "history": st["history"], "consensus": st.get("consensus", False),
                            "rounds": st["history"][-1]["round"], "docNotes": st.get("docNotes", []), "brief2": st["brief2"],
                            "rules2": st["rules2"], "cached": cached}, "run_write.js")


def cmd_status():
    st = load()
    print({k: st.get(k) for k in ("round", "stage", "overallRound", "consensus", "runs")})
    for h in st["history"][-3:]:
        n = len(h["agreeFrac"])
        print(f"  round {h['round']}: {n - len(h['weak'])}/{n} at >=80%, weak {h['weak']}, material {h.get('material')}")


if __name__ == "__main__":
    {"fix": cmd_fix, "merge-fix": cmd_merge_fix, "review": cmd_review, "merge-review": cmd_merge_review,
     "write": cmd_write, "status": cmd_status}[sys.argv[1]]()
