"""
Authentication module for ORIGIN SAR Oil Slick Detection web application.
Provides secure user credential management, password hashing, registration, and login verification.
"""

import os
import json
import hashlib
from pathlib import Path

# Path to local user database
USER_DB_PATH = Path("data/users.json")


def _hash_password(password: str) -> str:
    """Hashes a plain-text password using SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> dict:
    """
    Loads registered users from the local JSON database.
    Creates default admin and analyst accounts if the database does not exist.
    """
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not USER_DB_PATH.exists():
        default_users = {
            "admin": {
                "password_hash": _hash_password("admin123"),
                "full_name": "System Administrator",
                "role": "Administrator"
            },
            "analyst": {
                "password_hash": _hash_password("sar2026"),
                "full_name": "SAR Marine Analyst",
                "role": "Analyst"
            }
        }
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)
        return default_users

    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: dict) -> None:
    """Saves the updated users dictionary to the local JSON database."""
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def authenticate_user(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Verifies user credentials.

    Returns:
        tuple: (is_authenticated, message, user_info)
    """
    users = load_users()
    username_clean = username.strip().lower()

    if username_clean not in users:
        return False, "Username not found. Please check credentials or sign up.", {}

    user_data = users[username_clean]
    if user_data.get("password_hash") == _hash_password(password):
        return True, "Login successful!", user_data
    else:
        return False, "Incorrect password. Please try again.", {}


def register_user(username: str, password: str, full_name: str, role: str = "Analyst") -> tuple[bool, str]:
    """
    Registers a new user account.

    Returns:
        tuple: (success, message)
    """
    username_clean = username.strip().lower()

    if not username_clean or not password:
        return False, "Username and password cannot be empty."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    users = load_users()
    if username_clean in users:
        return False, f"Username '{username_clean}' already exists. Please choose a different username."

    users[username_clean] = {
        "password_hash": _hash_password(password),
        "full_name": full_name.strip() if full_name else username_clean.capitalize(),
        "role": role
    }
    save_users(users)
    return True, f"Account '{username_clean}' successfully created! You can now log in."
