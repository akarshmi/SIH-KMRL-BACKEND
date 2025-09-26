from passlib.context import CryptContext

# Create CryptContext with bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # optional: adjust hash strength
)

def hash_password(password: str) -> str:
    """
    Hash a password safely for storage.
    - Truncates to 72 bytes (bcrypt limit)
    - Encodes to UTF-8
    """
    # truncate to 72 bytes to avoid bcrypt limitation
    pw_bytes = password.encode("utf-8")[:72]
    return pwd_context.hash(pw_bytes)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    - Encodes password to UTF-8
    """
    return pwd_context.verify(plain_password.encode("utf-8")[:72], hashed_password)
