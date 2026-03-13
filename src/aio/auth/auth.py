import base64
import requests
from typing import Any, Dict

def get_api_key(auth_cfg: Dict[str, Any], client_id: str, client_secret: str,username: str, password: str) -> str:
    """Obtain bearer token using password grant
    auth_cfg must provide: token_url, token_scope.

    Get api_key using password grant, with client_id and client_secret for basic auth, 
    and username/password for resource owner credentials.
    """
    data_string = client_id + ":" + client_secret
    
    # Convert string to bytes
    data_bytes = data_string.encode('utf-8')
    # Encode bytes to base64
    encoded_bytes = base64.b64encode(data_bytes)
    # Convert the resulting bytes back to a string
    basic_auth_token = encoded_bytes.decode('utf-8') if isinstance(encoded_bytes, bytes) else encoded_bytes

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {basic_auth_token}",
    }
    data = {
        "scope": auth_cfg["token_scope"],
        "grant_type": "password",
        "username": username,
        "password": password
    }
    resp = requests.post(auth_cfg["token_url"], headers=headers, data=data, timeout=120)
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError("Failed to obtain access_token from token endpoint")
    return token