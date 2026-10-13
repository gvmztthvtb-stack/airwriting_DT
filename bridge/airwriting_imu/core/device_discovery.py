from __future__ import annotations

import socket
from typing import Optional, Tuple

DISCOVERY_REQUEST = "AIRWRITING_DISCOVER_V1"
DISCOVERY_RESPONSE_PREFIX = "AIRWRITING_SERVER_V1"


def is_discovery_request(payload: str) -> bool:
    return payload.strip() == DISCOVERY_REQUEST


def build_discovery_response(server_ip: str, udp_port: int) -> str:
    return f"{DISCOVERY_RESPONSE_PREFIX} {server_ip} {int(udp_port)}"


def parse_discovery_response(payload: str) -> Optional[Tuple[str, int]]:
    parts = payload.strip().split()
    if len(parts) != 3 or parts[0] != DISCOVERY_RESPONSE_PREFIX:
        return None

    server_ip = parts[1].strip()
    try:
        socket.inet_aton(server_ip)
    except OSError:
        return None

    try:
        udp_port = int(parts[2])
    except ValueError:
        return None

    if not (0 < udp_port <= 65535):
        return None

    return server_ip, udp_port


def resolve_local_ip_for_peer(peer_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_host, 9))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()
