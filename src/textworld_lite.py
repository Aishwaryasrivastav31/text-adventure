"""
TextWorld-Lite: a small, procedurally generated, partially observable text adventure.

Goal: find the key that MATCHES the locked door (door text names its material, e.g.
"a locked brass door" -> "brass key"), unlock it, enter the vault, take the treasure.
Decoy keys of other materials exist, and the agent can carry only 2 items (drop to free space).
Once the door has been seen, a "Journal:" line keeps reminding the agent of its material (otherwise the
information would be unobservable in the key rooms - a memory problem, not a language problem).
The agent only sees TEXT (current-room description, inventory, last action) and picks
from a list of admissible text commands (the standard "command list" setting used by
TextWorld / Jericho benchmarks).

Vocabulary and phrasing are controlled by a `Lexicon`, which lets us build
held-out test splits with unseen words / unseen paraphrases to measure language
generalisation.
"""
import random
from dataclasses import dataclass, field
from typing import List

DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass
class Lexicon:
    name: str
    rooms: List[str]
    keys: List[str]
    treasures: List[str]
    distractors: List[str]
    t_room: List[str]
    t_exits: List[str]
    t_door: List[str]
    t_item: List[str]
    t_treasure: List[str]
    t_inv: List[str]
    t_empty: List[str]
    t_journal: List[str]
    vault_names: List[str] = field(default_factory=lambda: ["vault"])


TRAIN = Lexicon(
    name="train",
    rooms=["kitchen", "cellar", "library", "garden", "attic", "hall", "study"],
    keys=["brass key", "iron key", "silver key", "tin key"],
    treasures=["golden chalice", "silver goblet"],
    distractors=["lamp", "book", "apple", "coin"],
    t_room=["You are standing in the {room}.", "This is the {room}."],
    t_exits=["Exits lead {dirs}.", "You can go {dirs}."],
    t_door=["A locked {mat} door blocks the way {dir}."],
    t_item=["You see a {item} here.", "There is a {item} on the floor."],
    t_treasure=["A glittering treasure rests here: a {item}."],
    t_inv=["You are carrying: {items}."],
    t_empty=["Your hands are empty."],
    t_journal=["You recall a locked {mat} door."],
)

# Split (b): same phrasing, but unseen room / object names.
NEW_NAMES = Lexicon(
    name="new_names",
    rooms=["pantry", "balcony", "workshop", "chapel", "armory"],
    keys=["copper key", "bronze key", "steel key", "zinc key"],
    treasures=["jeweled crown", "emerald amulet"],
    distractors=["candle", "scroll", "pebble", "feather"],
    t_room=TRAIN.t_room, t_exits=TRAIN.t_exits, t_door=TRAIN.t_door,
    t_item=TRAIN.t_item, t_treasure=TRAIN.t_treasure, t_inv=TRAIN.t_inv,
    t_empty=TRAIN.t_empty, t_journal=TRAIN.t_journal,
)

# Split (c): unseen names AND unseen paraphrases of every template.
PARAPHRASE = Lexicon(
    name="paraphrase",
    rooms=NEW_NAMES.rooms, keys=NEW_NAMES.keys, treasures=NEW_NAMES.treasures,
    distractors=NEW_NAMES.distractors,
    t_room=["You find yourself inside the {room}.", "Around you is the {room}."],
    t_exits=["Passages open to the {dirs}.", "Ways out: {dirs}."],
    t_door=["A bolted {mat} door bars the path {dir}."],
    t_item=["Lying nearby: a {item}.", "Close at hand you notice a {item}."],
    t_treasure=["A dazzling prize is displayed here: a {item}."],
    t_inv=["You hold: {items}."],
    t_empty=["You carry nothing."],
    t_journal=["You remember a bolted {mat} door."],
)

# Validation-only unseen-names lexicon (different words from NEW_NAMES, so tuning never sees test words).
VAL_NAMES = Lexicon(
    name="val_names",
    rooms=["cabin", "greenhouse", "dungeon", "nursery", "observatory"],
    keys=["gold key", "lead key", "nickel key", "chrome key"],
    treasures=["silver tiara", "amber orb"],
    distractors=["bottle", "ribbon", "spoon", "mirror"],
    t_room=TRAIN.t_room, t_exits=TRAIN.t_exits, t_door=TRAIN.t_door,
    t_item=TRAIN.t_item, t_treasure=TRAIN.t_treasure, t_inv=TRAIN.t_inv,
    t_empty=TRAIN.t_empty, t_journal=TRAIN.t_journal,
)

# Training-time vocabulary randomisation: the real training words plus many random pseudo-words
# (random syllable strings). Verified disjoint from every validation / test word list.
def _pseudo(n, seed, taken):
    rng = random.Random(seed); syl = ["ba", "ke", "to", "mi", "ra", "lo", "su", "ne", "vi", "do", "fa", "gu", "ze", "pi", "wo"]
    out = set()
    while len(out) < n:
        w = "".join(rng.choice(syl) for _ in range(rng.randint(2, 3)))
        if w not in taken:
            out.add(w)
    return sorted(out)

_HELD = set()
for _lx in (NEW_NAMES, VAL_NAMES, PARAPHRASE):
    for _pool in (_lx.rooms, _lx.keys, _lx.treasures, _lx.distractors):
        for _x in _pool:
            _HELD.update(_x.split())
_mats = _pseudo(60, 1, _HELD)
_rooms = _pseudo(60, 2, _HELD | set(_mats))
_dist = _pseudo(50, 3, _HELD | set(_mats) | set(_rooms))
_trs = _pseudo(40, 4, _HELD | set(_mats) | set(_rooms) | set(_dist))
TRAIN_AUG = Lexicon(
    name="train_aug",
    rooms=TRAIN.rooms + _rooms,
    keys=TRAIN.keys + [f"{m} key" for m in _mats],
    treasures=TRAIN.treasures + [f"{t} idol" for t in _trs],
    distractors=TRAIN.distractors + _dist,
    t_room=TRAIN.t_room, t_exits=TRAIN.t_exits, t_door=TRAIN.t_door,
    t_item=TRAIN.t_item, t_treasure=TRAIN.t_treasure, t_inv=TRAIN.t_inv,
    t_empty=TRAIN.t_empty, t_journal=TRAIN.t_journal,
)

LEXICONS = {"train": TRAIN, "train_aug": TRAIN_AUG, "new_names": NEW_NAMES, "paraphrase": PARAPHRASE, "val_names": VAL_NAMES}


class TextWorld:
    MAX_STEPS = 50
    CAPACITY = 2
    STEP_PENALTY = -0.05
    R_KEY, R_UNLOCK, R_TREASURE = 1.0, 1.0, 10.0

    def __init__(self, lexicon: Lexicon = TRAIN, n_rooms=(6, 8)):
        self.lex = lexicon
        self.n_rooms = n_rooms

    def reset(self, seed: int):
        rng = random.Random(seed)
        self.rng = rng
        n = rng.randint(*self.n_rooms)
        pos = {0: (0, 0)}
        adj = {0: {}}
        for i in range(1, n):
            while True:
                p = rng.choice(list(pos))
                d = rng.choice(list(DIRS))
                if d in adj[p]:
                    continue
                q = (pos[p][0] + DIRS[d][0], pos[p][1] + DIRS[d][1])
                if q in pos.values():
                    continue
                pos[i] = q
                adj[i] = {OPP[d]: p}
                adj[p][d] = i
                break
        leaves = [r for r in adj if len(adj[r]) == 1 and r != 0]
        self.vault = rng.choice(leaves)
        (self.lock_dir, self.lock_nbr), = adj[self.vault].items()   # dir FROM vault -> neighbour
        self.adj = adj
        names = iter(rng.sample(self.lex.rooms * 2, n - 1))
        self.names = {r: ("vault" if r == self.vault else next(names)) for r in adj}
        self.key = rng.choice(self.lex.keys)
        self.treasure = rng.choice(self.lex.treasures)
        non_vault = [r for r in adj if r != self.vault]
        self.items = {r: [] for r in adj}
        self.items[rng.choice(non_vault)].append(self.key)
        for k in rng.sample([k for k in self.lex.keys if k != self.key], rng.randint(1, 2)):
            self.items[rng.choice(non_vault)].append(k)       # decoy keys
        self.items[self.vault].append(self.treasure)
        for d in rng.sample(self.lex.distractors, rng.randint(2, 3)):
            self.items[rng.choice(non_vault)].append(d)
        self.loc = rng.choice(non_vault)
        self.inv, self.unlocked, self.got_key, self.seen_door = [], False, False, False
        self.steps, self.done = 0, False
        self.hist, self.feedback = [], ""
        return self.observation()

    def door_dir(self):
        """Direction from the vault's neighbour towards the vault."""
        return OPP[self.lock_dir]

    def observation(self) -> str:
        L, rng = self.lex, self.rng
        parts = [rng.choice(L.t_room).format(room=self.names[self.loc])]
        parts.append(rng.choice(L.t_exits).format(dirs=", ".join(sorted(self.adj[self.loc]))))
        if self.loc == self.lock_nbr and not self.unlocked:
            parts.append(rng.choice(L.t_door).format(dir=self.door_dir(), mat=self.key.split()[0]))
        for it in self.items[self.loc]:
            tpl = L.t_treasure if it == self.treasure else L.t_item
            parts.append(rng.choice(tpl).format(item=it))
        parts.append(rng.choice(L.t_inv).format(items=", ".join(self.inv)) if self.inv
                     else rng.choice(L.t_empty))
        if self.feedback:
            parts.append(self.feedback)
        if self.loc == self.lock_nbr:
            self.seen_door = True
        if self.seen_door and not self.unlocked:       # persistent memory of the door text
            parts.append("Journal: " + rng.choice(L.t_journal).format(mat=self.key.split()[0]))
        parts.append("Recent actions: " + (", ".join(self.hist[-3:]) if self.hist else "none") + ".")
        return " ".join(parts)

    def admissible(self) -> List[str]:
        cmds = [f"go {d}" for d in sorted(self.adj[self.loc])]
        cmds += [f"take {i}" for i in self.items[self.loc]]
        if self.loc == self.lock_nbr and not self.unlocked and self.key in self.inv:
            cmds.append(f"unlock {self.door_dir()} door with {self.key}")
        cmds += [f"drop {i}" for i in self.inv]
        cmds.append("look")
        return cmds

    def step(self, cmd: str):
        assert not self.done, "episode over"
        r = self.STEP_PENALTY
        self.feedback = ""
        if cmd.startswith("go "):
            tgt = self.adj[self.loc].get(cmd[3:])
            if tgt == self.vault and not self.unlocked:
                self.feedback = "The door is locked."
            elif tgt is not None:
                self.loc = tgt
        elif cmd.startswith("take "):
            it = cmd[5:]
            if it in self.items[self.loc] and len(self.inv) >= self.CAPACITY:
                self.feedback = "Your hands are full."
            elif it in self.items[self.loc]:
                self.items[self.loc].remove(it)
                self.inv.append(it)
                if it == self.key and not self.got_key:
                    self.got_key = True
                    r += self.R_KEY
                if it == self.treasure:
                    r += self.R_TREASURE
                    self.done = True
        elif cmd.startswith("drop "):
            it = cmd[5:]
            if it in self.inv:
                self.inv.remove(it)
                self.items[self.loc].append(it)
        elif cmd.startswith("unlock ") and cmd in self.admissible():
            self.unlocked = True
            r += self.R_UNLOCK
            self.feedback = "Click! The door swings open."
        self.hist.append(cmd)
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True
        return self.observation(), r, self.done, {"success": self.treasure in self.inv}
