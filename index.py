"""
api/index.py
-------------
Self-contained CCP backend for Vercel serverless deployment.

All logic (preference model, solver, crypto) is inlined here so
Vercel can bundle it as a single serverless function with no import
path issues.

Architecture decision: the API is stateless. The frontend holds
commitments in the browser session and sends them all with each
solve request. This is correct for a demo and requires zero database.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import json
import secrets
import time

app = FastAPI(title="CCP — Conditional Commitment Protocol")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceInput(BaseModel):
    participant_id: str
    action: str
    contribution: float
    min_participants: int
    min_pool: float
    deadline: float        # unix timestamp
    confidence: float = 1.0


class CommitRequest(BaseModel):
    preference: PreferenceInput


class SolveRequest(BaseModel):
    preferences: List[PreferenceInput]


class VerifyAndSolveRequest(BaseModel):
    """
    Full commit-reveal-solve in one request for the live demo.
    Each entry carries the preference, nonce, and commitment hash
    so the server can verify honesty before solving.
    """
    entries: List[dict]   # [{preference, nonce, commitment_hash}]


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO (commit-reveal)
# ══════════════════════════════════════════════════════════════════════════════

def _pref_to_json(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def make_commitment(pref_dict: dict) -> dict:
    nonce = secrets.token_hex(16)
    payload = _pref_to_json(pref_dict) + nonce
    commitment_hash = hashlib.sha256(payload.encode()).hexdigest()
    return {"nonce": nonce, "commitment_hash": commitment_hash}


def verify_commitment(pref_dict: dict, nonce: str, commitment_hash: str) -> bool:
    payload = _pref_to_json(pref_dict) + nonce
    recomputed = hashlib.sha256(payload.encode()).hexdigest()
    return recomputed == commitment_hash


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER (maximal fixed-point pruning)
# ══════════════════════════════════════════════════════════════════════════════

def _pool_excluding(candidates: list, pid: str) -> float:
    return sum(p["contribution"] for p in candidates if p["participant_id"] != pid)


def solve(preferences: List[dict], now: float = None) -> dict:
    """
    Maximal fixed-point solver.

    Start with all active participants as candidates, repeatedly prune
    anyone whose conditions fail given the current group, until stable.
    Returns the maximal stable coalition.
    """
    now = now or time.time()
    log = []

    active = [p for p in preferences if now <= p["deadline"]]
    log.append(f"Active participants: {len(active)}/{len(preferences)}")

    if not active:
        return {
            "found": False, "committed": [], "total_pool": 0,
            "iterations": 0, "log": "\n".join(log)
        }

    candidates = list(active)
    log.append(f"Starting with full candidate set: {len(candidates)} participants")

    iterations = 0
    while candidates:
        iterations += 1
        next_candidates = []
        removed = []

        for p in candidates:
            others = len(candidates) - 1
            pool_excl = _pool_excluding(candidates, p["participant_id"])
            if others >= p["min_participants"] and pool_excl >= p["min_pool"]:
                next_candidates.append(p)
            else:
                removed.append(p["participant_id"])

        if removed:
            log.append(f"Iteration {iterations}: removed {removed} → {len(next_candidates)} remaining")
        else:
            log.append(f"Iteration {iterations}: no removals — fixed point reached")

        if len(next_candidates) == len(candidates):
            candidates = next_candidates
            break
        candidates = next_candidates

    if candidates:
        total = sum(p["contribution"] for p in candidates)
        log.append(f"Equilibrium found: {len(candidates)} participants, pool = {total}")
        return {
            "found": True,
            "committed": [{"id": p["participant_id"], "contribution": p["contribution"]} for p in candidates],
            "total_pool": total,
            "iterations": iterations,
            "log": "\n".join(log),
        }
    else:
        log.append("No stable equilibrium. Nobody is bound. Zero risk.")
        return {
            "found": False, "committed": [], "total_pool": 0,
            "iterations": iterations, "log": "\n".join(log),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CCP Solver API"}


@app.post("/api/commit")
def commit_endpoint(req: CommitRequest):
    """
    Phase 1: seal a preference function.
    Returns commitment_hash + nonce to the client.
    Client stores both; only the hash is "public" at this stage.
    """
    pref_dict = req.preference.model_dump()
    result = make_commitment(pref_dict)
    return {
        "participant_id": req.preference.participant_id,
        "commitment_hash": result["commitment_hash"],
        "commitment_hash_short": result["commitment_hash"][:16] + "...",
        "nonce": result["nonce"],   # client holds this privately until reveal
    }


@app.post("/api/solve")
def solve_endpoint(req: SolveRequest):
    """
    Direct solve (no commit-reveal): send all preference functions,
    get equilibrium result. Used for the simple demo flow.
    """
    prefs = [p.model_dump() for p in req.preferences]
    return solve(prefs)


@app.post("/api/verify-and-solve")
def verify_and_solve(req: VerifyAndSolveRequest):
    """
    Full commit-reveal-solve:
    Each entry contains {preference, nonce, commitment_hash}.
    Server verifies each reveal before solving.
    Rejected reveals are excluded from the solver.
    """
    verified = []
    verification_log = []

    for entry in req.entries:
        pref = entry["preference"]
        nonce = entry["nonce"]
        h = entry["commitment_hash"]
        ok = verify_commitment(pref, nonce, h)
        verification_log.append({
            "participant_id": pref["participant_id"],
            "valid": ok,
        })
        if ok:
            verified.append(pref)

    result = solve(verified)
    result["verification_log"] = verification_log
    return result
