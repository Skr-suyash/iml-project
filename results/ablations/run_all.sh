#!/usr/bin/env bash
# Overnight batch: every follow-up analysis, sequentially, each with its own log.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
L=results/ablations/logs
set -x
$PY -m vcascade.ablations signals                         > $L/signals.log 2>&1
$PY -m vcascade.ablations ablate    --ratios 5,20,60       > $L/ablate.log 2>&1
$PY -m vcascade.ablations trainsize --ratios 20,60         > $L/trainsize.log 2>&1
$PY -m vcascade.ablations shift     --ratios 20,60         > $L/shift.log 2>&1
$PY -m vcascade.experiment --cache cache --out results/token_cost --cost-model token --ratios 5,20,60 > $L/token.log 2>&1
$PY -m vcascade.ablations bootstrap --ratios 3,5,10,20,60,200 > $L/bootstrap.log 2>&1
$PY -m vcascade.ablations exact     --ratios 5,20,60 --splits 3 --n 400 > $L/exact.log 2>&1
echo ALL_DONE
# Follow-ups: training without gold labels, and the hidden-state probe (needs cache/hidden.npz)
$PY -m vcascade.ablations labelfree --cost-model unit  > $L/labelfree_unit.log 2>&1
$PY -m vcascade.ablations labelfree --cost-model token > $L/labelfree_token.log 2>&1
$PY -m vcascade.ablations probe     --cost-model unit  > $L/probe_unit.log 2>&1
$PY -m vcascade.ablations probe     --cost-model token > $L/probe_token.log 2>&1
echo FOLLOWUPS_DONE
