from fastapi.testclient import TestClient

from engine.modules.auth import AuthStore
from engine.server.app import create_app


def test_register_login_logout_and_password_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = AuthStore(tmp_path / "auth.sqlite3")
    client = TestClient(create_app(auth_store=store, auth_required=True))

    registered = client.post(
        "/api/auth/register",
        json={"email": "Owner@Example.com", "name": "Owner", "password": "password-123"},
    )
    assert registered.status_code == 200
    token = registered.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=headers).json()["user"]["email"] == "owner@example.com"
    assert client.get("/api/system/status").status_code == 200
    anonymous = TestClient(create_app(auth_store=store, auth_required=True))
    assert anonymous.get("/api/system/status").status_code == 401
    assert client.get("/api/system/status", headers=headers).status_code == 200

    forgot = client.post("/api/auth/forgot-password", json={"email": "owner@example.com"})
    reset_token = forgot.json()["reset_token"]
    assert client.post(
        "/api/auth/reset-password", json={"token": reset_token, "password": "new-password-456"}
    ).status_code == 200
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert client.post(
        "/api/auth/login", json={"email": "owner@example.com", "password": "new-password-456"}
    ).status_code == 200
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_duplicate_registration_and_generic_forgot_response(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(auth_store=AuthStore(tmp_path / "auth.sqlite3"), auth_required=True))
    body = {"email": "user@example.com", "name": "User", "password": "password-123"}
    assert client.post("/api/auth/register", json=body).status_code == 200
    assert client.post("/api/auth/register", json=body).status_code == 400
    response = client.post("/api/auth/forgot-password", json={"email": "missing@example.com"})
    assert response.status_code == 200
    assert "reset_token" not in response.json()
