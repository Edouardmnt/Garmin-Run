"""Le redéploiement envoie exactement la requête de « kubectl rollout restart »."""

import json

from scripts.redeploy import restart_request


def test_requete_de_redemarrage():
    req = restart_request("https://10.96.0.1:443", "garmin-run", "garmin-api", "jeton", "2026-10-05T03:30:00+00:00")
    assert req.full_url == "https://10.96.0.1:443/apis/apps/v1/namespaces/garmin-run/deployments/garmin-api"
    assert req.get_method() == "PATCH"
    assert req.headers["Authorization"] == "Bearer jeton"
    assert req.headers["Content-type"] == "application/strategic-merge-patch+json"
    annotations = json.loads(req.data)["spec"]["template"]["metadata"]["annotations"]
    assert annotations == {"kubectl.kubernetes.io/restartedAt": "2026-10-05T03:30:00+00:00"}
