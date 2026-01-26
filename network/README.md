### 🌍 Setting up a Planetary Relay Node (Bootstrap Node)

If you want to share Planetary Memory across the internet (WAN) instead of just your local network, you need to host a "Bootstrap Node". This acts as a meeting point for Farnsworth agents.

#### 1. Requirements
*   A VPS or server with a public IP (e.g., AWS EC2, DigitalOcean Droplet, or a Raspberry Pi with port forwarding).
*   Python 3.10+ installed.

#### 2. Setup
Upload the `network/bootstrap_node.py` script to your server.

```bash
# On your server
mkdir farnsworth-relay
# (Copy bootstrap_node.py here)
pip install websockets
```

#### 3. Run the Node
```bash
python bootstrap_node.py --port 8888
```

You should see: `🚀 Starting Bootstrap Node on 0.0.0.0:8888`.

#### 4. Connect Your Agents
On your local Farnsworth machine (and your friends' machines), edit the `.env` file created by the setup wizard:

```ini
# .env
ENABLE_PLANETARY_MEMORY=true
PLANETARY_USE_P2P=true
# Add this line:
FARNSWORTH_BOOTSTRAP_PEER=ws://<YOUR_SERVER_IP>:8888
```

Restart Farnsworth. Your local agent will now introduce itself to the Bootstrap Node and can exchange Skill Vectors with anyone else connected to it!

> **Security Note**: The current bootstrap node is public. Anyone knowing the IP can join. For private networks, use a VPN or add authentication logic to `bootstrap_node.py`.
