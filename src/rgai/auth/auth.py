import requests
from typing import Any, Dict

def get_bearer_token(auth_cfg: Dict[str, Any], username: str, password: str) -> str:
    """Obtain bearer token using password grant, mirroring data_agent_lite behavior.

    auth_cfg must provide: token_url, token_scope, token_basic_auth
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth_cfg['token_basic_auth']}",
    }
    data = {
        "scope": auth_cfg["token_scope"],
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    resp = requests.post(auth_cfg["token_url"], headers=headers, data=data, timeout=120)
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError("Failed to obtain access_token from token endpoint")
    return token