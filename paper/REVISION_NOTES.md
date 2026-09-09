# Revision notes (plan → manuscript)

Implements `/opt/cursor/artifacts/plans/wip_paper_revision_list_80d00c7b.plan.md` in `paper/`.

## Deliverables

| File | Role |
|------|------|
| [`wip_joint_hierarchical_idk.tex`](wip_joint_hierarchical_idk.tex) | Revised 4-page WIP body |
| [`citations.bib`](citations.bib) | Core RTS IDK + algorithmic + sibling WIP refs |
| [`LOCKED_NUMBERS.md`](LOCKED_NUMBERS.md) | Frozen claim + bakeoff + registry provenance |
| [`figures/`](figures/) | Hierarchy schematic, GA schematic, Acc-floor cost bars |

## Plan checklist

- [x] **Claim lock** — joint \((S,H)\) under Acc floor; no invent-IDK / invent-hierarchy / global-optimum claims
- [x] **Bib rebuild** — Wang, Abdelzaher RTS’23, RTNS’21, robust TECS, Timely RTSS’25, Chellapilla, Markatopoulou, RestoreML, Moscato, M3N note, Nguyen+Katikaneni, fault-tolerant RTAS
- [x] **Abstract/Intro** — CPS open → Timely fixed-\(H\) gap → sibling skip positioning → numeric preview
- [x] **Methods trim** — data+train one block; SA one block; GA compact; \(H_i\) vs \(\alpha\) vs \(S\) vs \(\bar{C}\)
- [x] **Table IV hygiene** — cite Timely Table IV for roles; report calibrated \(H_i\) from registry
- [x] **Results** — Findings → Experiments and Results; filled Fig.+Table; Jetson deferred (not measured here)
- [x] **Related Work + Conclusion** — short positioning; limitations + future (SoC, history, scene)

## Intentional omissions / honesty

- Jetson Nano live cascade numbers are **not** claimed (repo baselines are CPU Windows).
- Acc-floor bakeoff rows come from the team joint-opt protocol documented in `LOCKED_NUMBERS.md` (not re-run in this cloud checkout).
- M3N-VC cited via Zenodo + RestoreML (`m3nvc2025zenodo`, `li2025restoreml`) per the dataset README citation request.
