# glm-box remote access

The box has two deliberate SSH endpoints over Tailscale:

| Alias | Shell | Purpose |
|---|---|---|
| `glm-box-wsl` | Arch WSL, `root` | Normal builds, inference, downloads, benchmarks, `tmux` |
| `glm-box` | Windows, `PC` | Git pull and recovery when WSL is unavailable |

Use `glm-box-wsl` for ordinary work.  It is a real OpenSSH server inside Arch;
commands no longer need nested `wsl.exe`, transient systemd units, or one
Windows Scheduled Task per job.

## Normal use

```powershell
ssh glm-box-wsl
ssh -t glm-box-wsl tmux new-session -A -s insignia
scp .\some-file glm-box-wsl:/var/lib/insignia/
```

Detached work survives SSH disconnects:

```powershell
ssh glm-box-wsl tmux new-session -d -s my-job bash /absolute/path/to/job.sh
ssh glm-box-wsl tmux list-sessions
ssh -t glm-box-wsl tmux attach-session -t my-job
```

Do not run model-scale jobs through Windows OpenSSH.  Do not create per-job
Scheduled Tasks.  Start them under `tmux` on `glm-box-wsl`.

## Repository workflow

The authoritative checkout remains the Windows Git worktree at
`C:\coding\Insignia-glm53-dflash2`.  Its `.git` file contains a Windows path,
so update it through the Windows control endpoint, then build and run through
Arch:

```powershell
git push origin glm53-dflash2-4070ti-super
ssh glm-box git.exe -C C:/coding/Insignia-glm53-dflash2 pull --ff-only
ssh -t glm-box-wsl tmux new-session -A -s insignia
```

Inside Arch the checkout is
`/mnt/c/coding/Insignia-glm53-dflash2`.  Never use the stale remote
`C:\coding\Insignia` snapshot.

## Q3_K_XL download

The native launcher serializes Hugging Face downloads, locks out duplicate
runs, calculates remaining bytes with an HF dry run, and checks both the
physical Windows `C:` capacity and ext4 capacity.  While downloading it aborts
before either filesystem crosses its reserve.

```powershell
ssh glm-box-wsl bash /mnt/c/coding/Insignia-glm53-dflash2/ops/glm-box-q3-download.sh --preflight
ssh glm-box-wsl tmux new-session -d -s q3-k-xl-download bash /mnt/c/coding/Insignia-glm53-dflash2/ops/glm-box-q3-download.sh
ssh glm-box-wsl tail -n 40 /var/lib/insignia/q3-k-xl-download.log
```

Defaults are a 100 GiB host reserve, 64 GiB guest reserve, and 32 GiB temporary
reconstruction margin.  Override only for a measured reason with
`INSIGNIA_Q3_HOST_RESERVE_GIB`, `INSIGNIA_Q3_GUEST_RESERVE_GIB`, or
`INSIGNIA_Q3_TEMP_MARGIN_GIB`.

## How the endpoint is wired

```text
Tailscale MagicDNS desktop-hlvh09q:2222
  -> Windows portproxy 100.102.208.13:2222
  -> 127.0.0.1:2223
  -> Arch sshd 0.0.0.0:2223
```

- The Windows firewall admits port 2222 only on the Tailscale address and only
  from `100.64.0.0/10`.
- Arch accepts the project Ed25519 key only; root password and keyboard
  authentication are disabled.
- Trusted Arch host-key fingerprint:
  `SHA256:OMsOTt0r9HFPxXlwARYd0OcV1mZmxWbTO1uvNfwBas8`.
- `C:\Users\PC\.wslconfig` keeps localhost forwarding enabled and sets the
  WSL VM idle timeout to the Windows maximum.
- One machine-lifecycle task, `Insignia Arch SSH Bootstrap`, starts
  `wsl.exe -d Arch -u root -- /usr/bin/sleep infinity` at Windows logon.  This
  keeps the WSL VM alive; it does not encode or supervise project jobs.

The lifecycle task is necessary because a systemd service alone does not keep
a WSL distribution running.  See Microsoft's [systemd behavior](https://learn.microsoft.com/en-us/windows/wsl/systemd)
and [.wslconfig reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

## Recovery

If `glm-box-wsl` is unavailable, the Windows endpoint remains independent:

```powershell
ssh glm-box whoami
ssh glm-box schtasks.exe /Query /TN "\Insignia Arch SSH Bootstrap" /V /FO LIST
ssh glm-box schtasks.exe /Run /TN "\Insignia Arch SSH Bootstrap"
```

Then test `ssh glm-box-wsl true`.  Use Windows recovery only for WSL lifecycle,
port-proxy, firewall, disk, or Git-control work.
