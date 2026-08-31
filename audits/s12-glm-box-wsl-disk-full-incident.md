# Incident: glm-box WSL backing disk exhaustion

Date: 2026-08-31
Host: `glm-box` / `DESKTOP-HLVH09Q`
Severity: SEV-2 (inference and model-download environment unavailable; no detected data loss)
Status: Recovered and closed

## Summary

The resumable `UD-Q3_K_XL` Hugging Face download expanded Arch's dynamic
`C:\coding\ext4.vhdx` until the Windows `C:` volume had effectively no free
space.  Linux still reported hundreds of GiB free inside its nominal 1 TiB
ext4 filesystem, but the Windows volume backing the dynamically expanding VHD
could no longer allocate physical clusters.  The virtual block device then
rejected reads and writes, ext4 aborted its journal, and WSL remounted the root
filesystem read-only.

This was not a GPU OOM, host-RAM OOM, physical-NVMe health failure, or workload
contention event.  The Samsung 990 EVO Plus reported `Healthy`/`OK`; the direct
trigger was exhaustion of free space on the Windows backing volume.

## Impact

- The Q3_K_XL download stopped during Xet file reconstruction.
- Arch became read-only and subsequently failed to start.
- Inference and benchmarks on glm-box were unavailable during recovery.
- No benchmark result was collected while the download or recovery I/O was
  active.
- No confirmed model or repository data was lost.

## Detection and evidence

The download terminated with:

```text
RuntimeError: File reconstruction error: Internal Writer Error:
Background writer channel closed
OSError: [Errno 30] Read-only file system
```

The kernel journal at `2026-08-31 21:26:26` showed the actual lower-level
failure:

```text
sd 0:0:0:3: rejecting I/O to offline device
I/O error, dev sdd
Aborting journal on device sdd-8
EXT4-fs (sdd): I/O error while writing superblock
EXT4-fs (sdd): Remounting filesystem read-only
```

Windows measurements immediately after the failure:

```text
C: free:             8.96 MiB (later samples: 44-46 MiB)
Physical disk:       Samsung SSD 990 EVO Plus 1TB
Health/operation:    Healthy / OK
Arch VHD path:       C:\coding\ext4.vhdx
VHD physical length: 662,358,196,224 bytes
```

After repair, Arch reported 615 GiB used and 342 GiB free inside the 1 TiB
ext4 filesystem.  That apparent free space was not backed by equivalent free
space on `C:` and therefore was not a safe download budget.

The partial Q3 tree occupied 94 GiB:

- 92 GiB in `.cache/huggingface/download/UD-Q3_K_XL`;
- one completed 2.1 GiB fourth shard;
- three approximately 31 GiB Xet reconstruction fragments.

## Root cause

The capacity gate used Linux `df` inside WSL as if it represented physical
capacity.  WSL's dynamic VHD exposes a nominal 1 TiB ext4 filesystem even
though the host `C:` volume is only about 930.6 GiB and also contains Windows,
hibernation, the pagefile, repositories, and other files.  Growing the VHD to
service Q3 staging exhausted NTFS first.

The filesystem failure was a consequence of the host allocation failure, not
evidence that the SSD media itself failed.

## Contributing factors

1. No Windows-host free-space watermark guarded large WSL downloads.
2. Xet retained three large partial reconstruction fragments from interrupted
   attempts, increasing peak working-set space.
3. The 27.4 GB hibernation file and 6.2 GB pagefile correctly consumed host
   reserve, but that reserve was not included in download planning.
4. A superseded 150.770 GiB XPR1-v1 expert sidecar remained beside its validated
   XPR1-v2 replacement.
5. The VHD grows automatically but does not automatically compact after Linux
   files are removed.

## Recovery procedure

1. Stopped the failed Hugging Face process and all model I/O.
2. Verified physical NVMe and NTFS health from Windows.
3. Temporarily disabled hibernation, releasing 27.4 GB so recovery tools had
   host-side working space.  Hibernation was restored after compaction.
4. Restarted the wedged `WslService`; Windows and SSH remained online.
5. Followed Microsoft's offline WSL VHD repair procedure:
   - shut down WSL;
   - attached `C:\coding\ext4.vhdx` with `--vhd --bare`;
   - installed a temporary Debian repair distro because Arch was the only
     registered distro;
   - identified the unmounted target as `/dev/sdd`, 1 TiB ext4, UUID
     `515e7c7a-3a5c-4a98-80f1-3aeab1190518`;
   - ran a no-write `e2fsck -fn` audit;
   - applied only automatic safe repair with `e2fsck -fp`;
   - ran a second `e2fsck -fn`, which exited cleanly.
6. Verified Arch remounted read/write.
7. Revalidated the retained XPR1-v2 sidecar across all 12,096 records and all
   36,288 scale-prefix regions before deleting the superseded v1 sidecar.
8. Removed only the reproducible v1 sidecar, freeing 150.770 GiB inside ext4.
9. Ran `fstrim -v /`, which discarded 396.8 GiB of free ext4 blocks.
10. Shut down WSL and compacted the detached dynamic VHD.  The first pass
    reduced it from 662.36 GB to 499.52 GB and raised host free space to
    174.64 GB.
11. With explicit user authorization, removed the now-obsolete NVFP4 compact
    store, its index, the validated XPR1-v2 sidecar, and the derived dense FP8
    cache.  DFlash2 checkpoints, source, audits, traces, and fixtures were
    preserved.
12. Ran a second trim (562.3 GiB discarded) and offline compaction, then
    restored hibernation and removed the temporary Debian repair distro.
13. Installed a guarded native Q3 resume launcher that measures physical host
    capacity through `/mnt/c`, computes remaining Hub bytes before starting,
    and continuously enforces host and guest reserves.

Microsoft procedure:
<https://learn.microsoft.com/en-us/windows/wsl/disk-space#how-to-repair-a-vhd-mounting-error>

## Integrity results

The pre-repair no-write check found only journal bookkeeping damage:

```text
Free blocks count wrong by 2,048 blocks
Free inodes count wrong by one inode
orphan_present set while orphan file was clean
```

It found no directory-connectivity, inode-reference, or block-ownership
failure.  `e2fsck -fp` replayed the journal and corrected the counts.  The
second full check exited zero.

The XPR1-v2 validator subsequently reported:

```text
IG53XPK1-v2 PASS: 12096 records, 162126811136 bytes,
all headers/directories, prefix recounts=36288 (all)
```

## User-authorized storage transition

Q3_K_XL is now the intended model on glm-box.  The user authorized deletion of
the remaining NVFP4 model assets after the first VHD compaction pass.  The
completed purge was limited to:

- the 180 GiB compact NVFP4 text store and its index;
- the validated 150.992 GiB XPR1-v2 expert sidecar;
- the 8.13 GiB dense FP8 cache derived for that NVFP4 engine.

DFlash2 checkpoints, source code, audits, benchmark evidence, and small kernel
fixtures were not part of that deletion.

## Prevention actions

1. Gate every large WSL download on **Windows `C:` free space**, not just the
   root ext4 capacity.  `/mnt/c` reports the physical backing-volume limit
   without Windows-process interop.
2. Reserve at least 100 GiB of Windows-host free space during model downloads;
   abort before crossing the reserve.
3. Include hibernation, pagefile, final-file size, and temporary reconstruction
   size in peak-space calculations.
4. Download huge Hub shards serially and verify/remove completed staging data
   between shards when possible.
5. Record both NTFS free space and ext4 free space in the download log.
6. Trim and offline-compact the VHD after deleting model-scale artifacts.
7. Never start inference benchmarks while a model download, trim, repair, or
   compaction is active.

## Final state

Measurements after the second compaction and hibernation restoration:

```text
Windows C: free:     511,844,757,504 bytes (476.7 GiB)
Arch VHD length:     135,119,503,360 bytes (125.8 GiB)
Arch ext4 available: 892,628,439,040 bytes (831.3 GiB)
Arch ext4 use:       14%
Hibernate available: yes
```

The VHD is read/write, `sshd` is active, and the independent Windows endpoint
remains available for recovery.  The Q3 tree is resumable at 94 GiB; one shard
is complete and the HF dry run reports 145.1 GB remaining.  Future runs use
`ops/glm-box-q3-download.sh`, whose 100 GiB host reserve, 64 GiB guest reserve,
and 32 GiB reconstruction margin fail before either backing filesystem is
exhausted.
