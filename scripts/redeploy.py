"""Redémarre l'API et le tableau de bord pour qu'ils récupèrent la dernière image publiée par la CI.

Équivalent de « kubectl rollout restart », sans kubectl : appel direct à l'API de Kubernetes,
avec le jeton du compte de service monté dans le Pod (droits limités à ces deux Deployments).
"""

import json
import os
import ssl
import urllib.request
from datetime import datetime, timezone

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
DEPLOYMENTS = os.getenv("REDEPLOY_TARGETS", "garmin-api,garmin-dashboard").split(",")


def restart_request(api_server: str, namespace: str, name: str, token: str, now: str) -> urllib.request.Request:
    """Requête PATCH qui modifie une annotation du modèle de Pod : Kubernetes recrée alors les Pods."""
    patch = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}}
    return urllib.request.Request(
        f"{api_server}/apis/apps/v1/namespaces/{namespace}/deployments/{name}",
        data=json.dumps(patch).encode(), method="PATCH",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/strategic-merge-patch+json"},
    )


def main() -> None:
    token = open(f"{SA_DIR}/token", encoding="utf-8").read().strip()
    namespace = open(f"{SA_DIR}/namespace", encoding="utf-8").read().strip()
    api_server = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_SERVICE_PORT']}"
    context = ssl.create_default_context(cafile=f"{SA_DIR}/ca.crt")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name in DEPLOYMENTS:
        with urllib.request.urlopen(restart_request(api_server, namespace, name, token, now), context=context) as r:
            print(f"{name} : redémarrage demandé (HTTP {r.status})")


if __name__ == "__main__":
    main()
