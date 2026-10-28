# Remote Lab PC Setup Guide

Follow these steps on the physical computer located in the lab. This ensures the machine never goes to sleep and is always securely accessible via SSH and Tailscale, even if the university/lab network has strict firewalls.

## Phase 1: Power & Sleep Settings (Critical)
You must prevent the computer from sleeping, otherwise its network card will power down and you will lose connection permanently.

**If the Lab PC is Windows:**
1. Open **Settings** > **System** > **Power & sleep**.
2. Set "Screen" to turn off after 10-15 minutes (optional, saves monitor life).
3. Set "Sleep" to **Never**.
4. Search for "Edit power plan" in the Windows Start menu -> Click **Change advanced power settings** -> Expand **Hard disk** -> **Turn off hard disk after** -> Set to **0** (Never).

**If the Lab PC is Linux (Ubuntu):**
1. Open **Settings** > **Power**.
2. Set "Blank Screen" to whatever you like.
3. Set "Automatic Suspend" to **Off**.

## Phase 2: Install Tailscale (The Mesh VPN)
Tailscale bypasses lab firewalls without needing to configure routers, and gives your PC a permanent private IP address.

1. Go to [tailscale.com](https://tailscale.com) and create a free account.
2. Download and install Tailscale for the lab PC's operating system.
3. Log in to your account through the app.
4. **Important**: Open Tailscale settings/preferences and ensure it is set to **Start with Windows/System on boot**.
5. Note the **Tailscale IP** assigned to this machine (it starts with `100.x.x.x`). You will need this later.

## Phase 3: Install and Enable OpenSSH Server
This allows you to remotely connect to the terminal of the PC.

**If the Lab PC is Windows:**
1. Open **Settings** > **Apps** > **Optional features**.
2. Click **Add a feature**, search for **OpenSSH Server**, and install it.
3. Open Windows PowerShell as **Administrator** and run these commands to start it and ensure it runs on boot automatically:
   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```

**If the Lab PC is Linux (Ubuntu):**
1. Open a terminal and run:
   ```bash
   sudo apt update
   sudo apt install openssh-server -y
   sudo systemctl enable ssh
   sudo systemctl start ssh
   ```

## Phase 4: Install Tmux (To survive disconnects)
Tmux creates a persistent terminal session. If your connection drops, the scripts inside Tmux keep running.

**If the Lab PC is Linux (Ubuntu):**
1. Run `sudo apt install tmux -y`.

**If the Lab PC is Windows:**
Windows doesn't natively support tmux well, but you can run your code inside WSL (Windows Subsystem for Linux).
1. Open your WSL terminal.
2. Run `sudo apt update && sudo apt install tmux -y`.

## Phase 5 (Optional but Recommended): AnyDesk Backdoor
As a failsafe GUI backdoor, install AnyDesk. This is just in case you ever need to click something on the screen.
1. Download AnyDesk and **fully install it** (do not just run the portable version).
2. Go to AnyDesk Settings > Security.
3. Check **"Enable unattended access"** and set a strong, unique password.
4. Ensure AnyDesk is set to start on system boot.
