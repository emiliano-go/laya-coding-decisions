# Training IO / memory plan (deferred)

Status: **deferred** (apply before the next training run).
Recorded: 2026-11-14, during the v6 head-only run (`out-m6-head`, job
`bgj-7vudl137`).

## Symptom

The main disk (root `/` + `/home`, `/dev/mapper/root`, nvme1n1) hits 100% IO
during training. It is not the dataset: data, tokenized cache, and checkpoints
already live on `/data` (`/dev/mapper/nvme_data`, nvme0n1).

## Cause (measured)

- `/swapfile` (15.6G, priority 10) is on the **root** btrfs; zram (31.3G,
  priority 100) is full.
- `vm.swappiness = 150`.
- During the run, training PID RSS was ~0.4G but **VmSwap = ~9.1G**, and
  `/swapfile` held ~10.8G.
- The tokenized items cache (`train_m6s...items.1024.256.pt`, 689M on disk)
  expands to ~9G of Python objects in `train_coding.py`. zram cannot hold it,
  so the overflow lives on the root swapfile and is paged in/out every epoch.
- Root filesystem is 89% full (420G/475G); `/data` has 183G free.

## Fix (next run, in order)

1. **Move file swap to `/data`** (requires sudo):
   ```sh
   # interim: new swap-outs go to /data, old file swap stays until freed
   sudo btrfs filesystem mkswapfile --size 16G /data/swapfile
   sudo swapon --priority 20 /data/swapfile
   # after the current run finishes:
   sudo swapoff /swapfile
   sudo rm /swapfile
   # update /etc/fstab: replace /swapfile with /data/swapfile
   ```
   Keep zram at priority 100. Verify with `swapon --show`.
2. **Stop loading the whole item cache into RAM**:
   - change `scripts/train_coding.py` to save/load the tokenized cache in
     shards (e.g. 20k items) and train shard-by-shard, freeing each one;
   - alternative: stream items from JSONL per epoch (lower memory, slower).
   - Target: RSS ~1-2G, `VmSwap` ~0.
3. **Keep every training artifact on `/data`**:
   - run from `/data/work/laya-coding-decisions` (copy or symlink), or set
     `TMPDIR=/data/tmp HF_HOME=/data/laya-ft/hf-cache XDG_CACHE_HOME=/data/cache
     TORCH_HOME=/data/torch` for the training process.
4. **Verify during the run**:
   - `grep VmSwap /proc/<pid>/status` ~0;
   - `iostat -x 1` shows nvme1n1 idle;
   - `swapon --show` shows `/data/swapfile`.

## `/data` space audit (2026-11-14)

`df`: 769G used / 183G free. Top consumers (du):

| Path | Size |
|---|---|
| `/data/kestrel-iso/release` | 313G (~59 ISO builds) |
| `/data/SteamLibrary` | 108G |
| `/data/models` | 90G |
| `/data/ISOs` | 58G |
| `/data/laya-ft` | 55G |
| `/data/ornith-byte` | 51G |
| `/data/winboat` | 40G |
| `/data/kestrel-kernel-build` | 29G |
| `/data/keel-perf` | 27G |
| `/data/kestrel-iso-build` | 23G |
| `/data/ai2140root` + `ai-2140.sfs` | 18G |
| `/data/kestrel-verify*` (4) | ~20G |
| `/data/rbg-semantic-venv` | 5.8G |

Cleanup candidates (need explicit approval, not done):
- `kestrel-iso/release`: keep latest 1-2 ISOs, delete the rest (~300G).
- `laya-ft`: delete `train_*.items.*.pt` except the active one; delete
  `out-*` except `out-multi`, `out-m5mem4-head`, `out-m6-head` (~20-25G).
- `kestrel-iso-build/cache` (20G), `kestrel-kernel-build/packaging` (29G),
  `keel-perf/tkldev` (23G), `kestrel-verify*` (20G).
- `/data/ISOs`: delete unneeded images.

## Related code

- `scripts/train_coding.py` (`load_items`, `make_batches`)
- `scripts/gen_dataset_v6.py` (`--base-limit` for smaller mixes)
- dsh-totem `src/laya.ts` (canonical state layout; unchanged by this plan)
