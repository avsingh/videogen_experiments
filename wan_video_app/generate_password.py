#!/usr/bin/env python3
"""
Script to generate password hash for Wan Video Generator
"""

from werkzeug.security import generate_password_hash
import getpass

print("=" * 50)
print("Wan Video Generator - Password Setup")
print("=" * 50)
print()

username = input("Enter username (default: admin): ").strip() or "admin"
password = getpass.getpass("Enter password: ")
password_confirm = getpass.getpass("Confirm password: ")

if password != password_confirm:
    print("❌ Passwords don't match!")
    exit(1)

if len(password) < 8:
    print("❌ Password must be at least 8 characters!")
    exit(1)

password_hash = generate_password_hash(password)

print()
print("✅ Password hash generated!")
print()
print("Add this to your wan_app.py file in the USERS dictionary:")
print()
print(f"    '{username}': '{password_hash}'")
print()
print("Example:")
print()
print("USERS = {")
print(f"    '{username}': '{password_hash}'")
print("}")
print()
